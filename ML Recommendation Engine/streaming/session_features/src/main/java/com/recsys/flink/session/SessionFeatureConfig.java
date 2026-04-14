package com.recsys.flink.session;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.configuration.TaskManagerOptions;

import java.time.Duration;

public class SessionFeatureConfig {

    private String kafkaBootstrapServers = "kafka:9092";
    private String kafkaTopic = "user-events";
    private String kafkaGroupId = "session-feature-consumer";

    private String redisHost = "redis:6379";
    private int redisPort = 6379;
    private String redisKeyPrefix = "sf:";
    private int redisTtlSeconds = 86400;

    private int sessionGapMinutes = 30;
    private int maxSessionDurationHours = 4;

    private int maxViewedItems = 50;
    private int maxSearchQueries = 20;

    private String checkpointDir = "s3://rec-system/checkpoints/session-features";
    private int checkpointIntervalMs = 60000;
    private int parallelism = 16;

    private int maxOutOfOrdernessSeconds = 120;

    private int lateEventWatermarkMinutes = 2;
    private int maxLateEventWindowMinutes = 60;

    public static SessionFeatureConfig fromEnvironment() {
        SessionFeatureConfig config = new SessionFeatureConfig();

        String kafkaBootstrap = System.getenv("KAFKA_BOOTSTRAP");
        if (kafkaBootstrap != null && !kafkaBootstrap.isEmpty()) {
            config.setKafkaBootstrapServers(kafkaBootstrap);
        }

        String redisHost = System.getenv("REDIS_HOST");
        if (redisHost != null && !redisHost.isEmpty()) {
            config.setRedisHost(redisHost);
        }

        String parallelism = System.getenv("PARALLELISM");
        if (parallelism != null && !parallelism.isEmpty()) {
            config.setParallelism(Integer.parseInt(parallelism));
        }

        String checkpointDir = System.getenv("CHECKPOINT_DIR");
        if (checkpointDir != null && !checkpointDir.isEmpty()) {
            config.setCheckpointDir(checkpointDir);
        }

        return config;
    }

    public Configuration toFlinkConfig() {
        Configuration config = new Configuration();

        config.setString("flink.checkpoint.dir", checkpointDir);
        config.setInteger("flink.checkpoint.interval", checkpointIntervalMs);
        config.setInteger("flink.parallelism.default", parallelism);

        config.setInteger(TaskManagerOptions.NUM_TASK_SLOTS, parallelism * 2);
        config.setString(TaskManagerOptions.MANAGED_MEMORY_SIZE, "3g");

        config.setInteger(RestOptions.PORT, 8081);

        return config;
    }

    public String getKafkaBootstrapServers() {
        return kafkaBootstrapServers;
    }

    public void setKafkaBootstrapServers(String kafkaBootstrapServers) {
        this.kafkaBootstrapServers = kafkaBootstrapServers;
    }

    public String getKafkaTopic() {
        return kafkaTopic;
    }

    public void setKafkaTopic(String kafkaTopic) {
        this.kafkaTopic = kafkaTopic;
    }

    public String getKafkaGroupId() {
        return kafkaGroupId;
    }

    public void setKafkaGroupId(String kafkaGroupId) {
        this.kafkaGroupId = kafkaGroupId;
    }

    public String getRedisHost() {
        return redisHost;
    }

    public void setRedisHost(String redisHost) {
        this.redisHost = redisHost;
    }

    public int getRedisPort() {
        return redisPort;
    }

    public void setRedisPort(int redisPort) {
        this.redisPort = redisPort;
    }

    public String getRedisKeyPrefix() {
        return redisKeyPrefix;
    }

    public void setRedisKeyPrefix(String redisKeyPrefix) {
        this.redisKeyPrefix = redisKeyPrefix;
    }

    public int getRedisTtlSeconds() {
        return redisTtlSeconds;
    }

    public void setRedisTtlSeconds(int redisTtlSeconds) {
        this.redisTtlSeconds = redisTtlSeconds;
    }

    public int getSessionGapMinutes() {
        return sessionGapMinutes;
    }

    public void setSessionGapMinutes(int sessionGapMinutes) {
        this.sessionGapMinutes = sessionGapMinutes;
    }

    public int getMaxSessionDurationHours() {
        return maxSessionDurationHours;
    }

    public void setMaxSessionDurationHours(int maxSessionDurationHours) {
        this.maxSessionDurationHours = maxSessionDurationHours;
    }

    public int getMaxViewedItems() {
        return maxViewedItems;
    }

    public void setMaxViewedItems(int maxViewedItems) {
        this.maxViewedItems = maxViewedItems;
    }

    public int getMaxSearchQueries() {
        return maxSearchQueries;
    }

    public void setMaxSearchQueries(int maxSearchQueries) {
        this.maxSearchQueries = maxSearchQueries;
    }

    public String getCheckpointDir() {
        return checkpointDir;
    }

    public void setCheckpointDir(String checkpointDir) {
        this.checkpointDir = checkpointDir;
    }

    public int getCheckpointIntervalMs() {
        return checkpointIntervalMs;
    }

    public void setCheckpointIntervalMs(int checkpointIntervalMs) {
        this.checkpointIntervalMs = checkpointIntervalMs;
    }

    public int getParallelism() {
        return parallelism;
    }

    public void setParallelism(int parallelism) {
        this.parallelism = parallelism;
    }

    public int getMaxOutOfOrdernessSeconds() {
        return maxOutOfOrdernessSeconds;
    }

    public void setMaxOutOfOrdernessSeconds(int maxOutOfOrdernessSeconds) {
        this.maxOutOfOrdernessSeconds = maxOutOfOrdernessSeconds;
    }

    public int getLateEventWatermarkMinutes() {
        return lateEventWatermarkMinutes;
    }

    public void setLateEventWatermarkMinutes(int lateEventWatermarkMinutes) {
        this.lateEventWatermarkMinutes = lateEventWatermarkMinutes;
    }

    public int getMaxLateEventWindowMinutes() {
        return maxLateEventWindowMinutes;
    }

    public void setMaxLateEventWindowMinutes(int maxLateEventWindowMinutes) {
        this.maxLateEventWindowMinutes = maxLateEventWindowMinutes;
    }

    public Duration getSessionGapDuration() {
        return Duration.ofMinutes(sessionGapMinutes);
    }

    public Duration getMaxOutOfOrdernessDuration() {
        return Duration.ofSeconds(maxOutOfOrdernessSeconds);
    }

    private String redisPassword = System.getenv().getOrDefault("REDIS_PASSWORD", "");
    public String getRedisPassword() { return redisPassword; }
}