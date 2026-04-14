package com.recsys.flink.session.sink;

import com.recsys.flink.session.SessionFeatureConfig;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;
import redis.clients.jedis.Pipeline;
import java.util.ArrayList;
import java.util.List;

public class RedisSink extends RichSinkFunction<String> {

    private static final Logger LOG = LoggerFactory.getLogger(RedisSink.class);

    private final SessionFeatureConfig config;
    private transient JedisPool jedisPool;
    private static final int BATCH_SIZE = 100;
    private final List<String[]> writeBuffer = new ArrayList<>();

    public RedisSink(SessionFeatureConfig config) {
        this.config = config;
    }

    @Override
    public void open(Configuration parameters) {
        JedisPoolConfig poolConfig = new JedisPoolConfig();
        poolConfig.setMaxTotal(20);
        poolConfig.setMaxIdle(10);
        poolConfig.setMinIdle(5);
        poolConfig.setTestOnBorrow(true);
        poolConfig.setTestWhileIdle(true);

        String redisPassword = config.getRedisPassword();
        if (redisPassword != null && !redisPassword.isEmpty()) {
            jedisPool = new JedisPool(poolConfig, config.getRedisHost(),
                                      config.getRedisPort(), 2000, redisPassword);
        } else {
            jedisPool = new JedisPool(poolConfig, config.getRedisHost(),
                                      config.getRedisPort());
        }

        LOG.info("Redis sink connected to {}:{}", config.getRedisHost(), config.getRedisPort());
    }

    @Override
    public void invoke(String value, Context context) {
        Map<String, Object> featureMap = parseFeatureString(value);
        String redisKey = (String) featureMap.get("redis_key");
        
        if (redisKey != null && !redisKey.isEmpty()) {
            featureMap.remove("redis_key");
            String jsonValue = toJsonString(featureMap);
            writeBuffer.add(new String[]{
                redisKey,
                String.valueOf(config.getRedisTtlSeconds()),
                jsonValue
            });
            if (writeBuffer.size() >= BATCH_SIZE) {
                flushBuffer();
            }
        } else {
            LOG.warn("No redis_key found in feature map, skipping write");
        }
    }

    private void flushBuffer() {
        if (writeBuffer.isEmpty()) return;
        try (Jedis jedis = jedisPool.getResource()) {
            Pipeline pipeline = jedis.pipelined();
            for (String[] entry : writeBuffer) {
                pipeline.setex(entry[0], Integer.parseInt(entry[1]), entry[2]);
            }
            pipeline.sync();
        } catch (Exception e) {
            LOG.error("Failed to flush Redis pipeline buffer", e);
        }
        writeBuffer.clear();
    }

    @Override
    public void close() {
        flushBuffer();
        if (jedisPool != null && !jedisPool.isClosed()) {
            jedisPool.close();
            LOG.info("Redis sink closed");
        }
    }

    private Map<String, Object> parseFeatureString(String value) {
        return (Map<String, Object>) (Map<?, ?>) value;
    }

    private String toJsonString(Map<String, Object> map) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        int count = 0;
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            if (count > 0) sb.append(", ");
            sb.append("\"").append(entry.getKey()).append("\": ");
            Object val = entry.getValue();
            if (val instanceof String) {
                sb.append("\"").append(val).append("\"");
            } else if (val instanceof Number) {
                sb.append(val);
            } else if (val instanceof Map) {
                sb.append(toJsonString((Map<String, Object>) val));
            } else if (val instanceof java.util.List) {
                sb.append("[");
                java.util.List<?> list = (java.util.List<?>) val;
                for (int i = 0; i < list.size(); i++) {
                    if (i > 0) sb.append(", ");
                    sb.append("\"").append(list.get(i)).append("\"");
                }
                sb.append("]");
            } else {
                sb.append("null");
            }
            count++;
        }
        sb.append("}");
        return sb.toString();
    }
}