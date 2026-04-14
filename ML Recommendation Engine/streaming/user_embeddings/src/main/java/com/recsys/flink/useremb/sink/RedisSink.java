package com.recsys.flink.useremb.sink;

import com.recsys.flink.useremb.UserEmbeddingConfig;
import com.recsys.flink.useremb.types.SessionUpdateEvent.UserEmbedding;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

import java.nio.ByteBuffer;
import java.util.Base64;

public class RedisSink extends RichSinkFunction<UserEmbedding> {

    private static final Logger LOG = LoggerFactory.getLogger(RedisSink.class);

    private final UserEmbeddingConfig config;
    private transient JedisPool jedisPool;

    public RedisSink(UserEmbeddingConfig config) {
        this.config = config;
    }

    @Override
    public void open(Configuration parameters) {
        JedisPoolConfig poolConfig = new JedisPoolConfig();
        poolConfig.setMaxTotal(20);
        poolConfig.setMaxIdle(10);

        String redisPassword = config.getRedisPassword();
        if (redisPassword != null && !redisPassword.isEmpty()) {
            jedisPool = new JedisPool(poolConfig, config.getRedisHost(),
                                      config.getRedisPort(), 2000, redisPassword);
        } else {
            jedisPool = new JedisPool(poolConfig, config.getRedisHost(),
                                      config.getRedisPort());
        }
        LOG.info("User Embedding Redis sink connected");
    }

    @Override
    public void invoke(UserEmbedding embedding, Context context) {
        try (Jedis jedis = jedisPool.getResource()) {
            String redisKey = config.getRedisKeyPrefix() + embedding.getUserId();
            byte[] embBytes = convertToBytes(embedding.getEmbedding());
            String base64Emb = Base64.getEncoder().encodeToString(embBytes);
            jedis.setex(redisKey, 86400, base64Emb);
            LOG.debug("Redis SET {} (embedding)", redisKey);
        } catch (Exception e) {
            LOG.error("Failed to write user embedding to Redis: {}", e.getMessage());
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() {
        if (jedisPool != null) jedisPool.close();
    }

    private byte[] convertToBytes(float[] embedding) {
        ByteBuffer buffer = ByteBuffer.allocate(embedding.length * 4);
        for (float v : embedding) {
            buffer.putFloat(v);
        }
        return buffer.array();
    }
}