package com.recsys.flink.itemstats.sink;

import com.recsys.flink.itemstats.ItemStatsConfig;
import com.recsys.flink.itemstats.types.ItemEvent.ItemStats;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

public class ItemStatsRedisSink extends RichSinkFunction<ItemStats> {

    private static final Logger LOG = LoggerFactory.getLogger(ItemStatsRedisSink.class);

    private final ItemStatsConfig config;
    private transient JedisPool jedisPool;

    public ItemStatsRedisSink(ItemStatsConfig config) {
        this.config = config;
    }

    @Override
    public void open(Configuration parameters) {
        JedisPoolConfig poolConfig = new JedisPoolConfig();
        poolConfig.setMaxTotal(20);
        poolConfig.setMaxIdle(10);
        poolConfig.setTestOnBorrow(true);

        String redisPassword = config.getRedisPassword();
        if (redisPassword != null && !redisPassword.isEmpty()) {
            jedisPool = new JedisPool(poolConfig, config.getRedisHost(),
                                      config.getRedisPort(), 2000, redisPassword);
        } else {
            jedisPool = new JedisPool(poolConfig, config.getRedisHost(),
                                      config.getRedisPort());
        }
        LOG.info("Item Stats Redis sink connected to {}:{}", config.getRedisHost(), config.getRedisPort());
    }

    @Override
    public void invoke(ItemStats stats, Context context) {
        try (Jedis jedis = jedisPool.getResource()) {
            String redisKey = config.getRedisKeyPrefix() + stats.getItemId();
            Map<String, Object> features = stats.toFeatureMap(
                    config.getRedisKeyPrefix(),
                    config.getCtrPriorClicks(),
                    config.getCtrPriorViews()
            );

            StringBuilder jsonBuilder = new StringBuilder("{");
            int count = 0;
            for (java.util.Map.Entry<String, Object> entry : features.entrySet()) {
                if ("redis_key".equals(entry.getKey())) continue;
                if (count > 0) jsonBuilder.append(", ");
                jsonBuilder.append("\"").append(entry.getKey()).append("\": ");
                Object val = entry.getValue();
                if (val instanceof Number) {
                    jsonBuilder.append(val);
                } else if (val instanceof String) {
                    jsonBuilder.append("\"").append(val).append("\"");
                } else {
                    jsonBuilder.append("null");
                }
                count++;
            }
            jsonBuilder.append("}");

            jedis.setex(redisKey, config.getRedisTtlSeconds(), jsonBuilder.toString());
            LOG.debug("Redis SET {} (TTL: {}s)", redisKey, config.getRedisTtlSeconds());
        } catch (Exception e) {
            LOG.error("Failed to write item stats to Redis: {}", e.getMessage());
            throw new RuntimeException("Item stats Redis write failed", e);
        }
    }

    @Override
    public void close() {
        if (jedisPool != null && !jedisPool.isClosed()) {
            jedisPool.close();
            LOG.info("Item Stats Redis sink closed");
        }
    }
}