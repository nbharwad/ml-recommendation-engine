package com.recsys.flink.useremb;

import org.apache.flink.configuration.Configuration;

import java.time.Duration;

public class UserEmbeddingConfig {

    private String kafkaBootstrapServers = "kafka:9092";
    private String kafkaTopic = "session-features";
    private String kafkaGroupId = "user-embedding-consumer";

    private String redisHost = "redis:6379";
    private int redisPort = 6379;
    private String redisKeyPrefix = "ue:";

    private String milvusHost = "milvus:19530";
    private String milvusCollection = "user_embeddings";

    private String modelPath = "/models/two_tower_user/model.onnx";
    private int embeddingDim = 128;

    private int windowSizeMinutes = 30;
    private int windowSlideMinutes = 5;

    private String checkpointDir = "s3://rec-system/checkpoints/user-embeddings";
    private int checkpointIntervalMs = 60000;
    private int parallelism = 16;

    private int minViewedItems = 3;

    public static UserEmbeddingConfig fromEnvironment() {
        UserEmbeddingConfig config = new UserEmbeddingConfig();
        
        String kafkaBootstrap = System.getenv("KAFKA_BOOTSTRAP");
        if (kafkaBootstrap != null && !kafkaBootstrap.isEmpty()) {
            config.setKafkaBootstrapServers(kafkaBootstrap);
        }

        String modelPath = System.getenv("MODEL_PATH");
        if (modelPath != null && !modelPath.isEmpty()) {
            config.setModelPath(modelPath);
        }

        return config;
    }

    public String getKafkaBootstrapServers() { return kafkaBootstrapServers; }
    public void setKafkaBootstrapServers(String kafkaBootstrapServers) { this.kafkaBootstrapServers = kafkaBootstrapServers; }

    public String getKafkaTopic() { return kafkaTopic; }
    public void setKafkaTopic(String kafkaTopic) { this.kafkaTopic = kafkaTopic; }

    public String getKafkaGroupId() { return kafkaGroupId; }
    public void setKafkaGroupId(String kafkaGroupId) { this.kafkaGroupId = kafkaGroupId; }

    public String getRedisHost() { return redisHost; }
    public void setRedisHost(String redisHost) { this.redisHost = redisHost; }

    public int getRedisPort() { return redisPort; }
    public void setRedisPort(int redisPort) { this.redisPort = redisPort; }

    public String getRedisKeyPrefix() { return redisKeyPrefix; }
    public void setRedisKeyPrefix(String redisKeyPrefix) { this.redisKeyPrefix = redisKeyPrefix; }

    public String getMilvusHost() { return milvusHost; }
    public void setMilvusHost(String milvusHost) { this.milvusHost = milvusHost; }

    public String getMilvusCollection() { return milvusCollection; }
    public void setMilvusCollection(String milvusCollection) { this.milvusCollection = milvusCollection; }

    public String getModelPath() { return modelPath; }
    public void setModelPath(String modelPath) { this.modelPath = modelPath; }

    public int getEmbeddingDim() { return embeddingDim; }
    public void setEmbeddingDim(int embeddingDim) { this.embeddingDim = embeddingDim; }

    public int getWindowSizeMinutes() { return windowSizeMinutes; }
    public void setWindowSizeMinutes(int windowSizeMinutes) { this.windowSizeMinutes = windowSizeMinutes; }

    public int getWindowSlideMinutes() { return windowSlideMinutes; }
    public void setWindowSlideMinutes(int windowSlideMinutes) { this.windowSlideMinutes = windowSlideMinutes; }

    public String getCheckpointDir() { return checkpointDir; }
    public void setCheckpointDir(String checkpointDir) { this.checkpointDir = checkpointDir; }

    public int getCheckpointIntervalMs() { return checkpointIntervalMs; }
    public void setCheckpointIntervalMs(int checkpointIntervalMs) { this.checkpointIntervalMs = checkpointIntervalMs; }

    public int getParallelism() { return parallelism; }
    public void setParallelism(int parallelism) { this.parallelism = parallelism; }

    public int getMinViewedItems() { return minViewedItems; }
    public void setMinViewedItems(int minViewedItems) { this.minViewedItems = minViewedItems; }

    public Duration getWindowSize() { return Duration.ofMinutes(windowSizeMinutes); }
    public Duration getWindowSlide() { return Duration.ofMinutes(windowSlideMinutes); }

    private String redisPassword = System.getenv().getOrDefault("REDIS_PASSWORD", "");
    public String getRedisPassword() { return redisPassword; }
}