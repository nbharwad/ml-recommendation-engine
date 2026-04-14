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
import redis.clients.jedis.Pipeline;

import java.nio.ByteBuffer;
import java.util.Base64;
import java.util.ArrayList;
import java.util.List;

public class RedisSink extends RichSinkFunction<UserEmbedding> {

    private static final Logger LOG = LoggerFactory.getLogger(RedisSink.class);

    private final UserEmbeddingConfig config;
    private transient JedisPool jedisPool;
    private static final int BATCH_SIZE = 100;
    private final List<String[]> writeBuffer = new ArrayList<>();

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
        String redisKey = config.getRedisKeyPrefix() + embedding.getUserId();
        byte[] embBytes = convertToBytes(embedding.getEmbedding());
        String base64Emb = Base64.getEncoder().encodeToString(embBytes);
        
        writeBuffer.add(new String[]{redisKey, "86400", base64Emb});
        if (writeBuffer.size() >= BATCH_SIZE) {
            flushBuffer();
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