package com.recsys.flink.itemstats;

import org.apache.flink.configuration.Configuration;

import java.time.Duration;

public class ItemStatsConfig {

    private String kafkaBootstrapServers = "kafka:9092";
    private String kafkaTopic = "user-events";
    private String kafkaGroupId = "item-stats-consumer";

    private String redisHost = "redis:6379";
    private int redisPort = 6379;
    private String redisKeyPrefix = "is:";
    private int redisTtlSeconds = 259200;

    private int windowSizeHours = 1;
    private int windowSlideMinutes = 5;

    private String checkpointDir = "s3://rec-system/checkpoints/item-stats";
    private int checkpointIntervalMs = 60000;
    private int parallelism = 32;

    private int maxOutOfOrdernessSeconds = 120;

    private double ctrPriorClicks = 5.0;
    private double ctrPriorViews = 100.0;

    public static ItemStatsConfig fromEnvironment() {
        ItemStatsConfig config = new ItemStatsConfig();

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

        return config;
    }

    public Configuration toFlinkConfig() {
        Configuration config = new Configuration();
        config.setString("flink.checkpoint.dir", checkpointDir);
        config.setInteger("flink.checkpoint.interval", checkpointIntervalMs);
        config.setInteger("flink.parallelism.default", parallelism);
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

    public int getRedisTtlSeconds() { return redisTtlSeconds; }
    public void setRedisTtlSeconds(int redisTtlSeconds) { this.redisTtlSeconds = redisTtlSeconds; }

    public int getWindowSizeHours() { return windowSizeHours; }
    public void setWindowSizeHours(int windowSizeHours) { this.windowSizeHours = windowSizeHours; }

    public int getWindowSlideMinutes() { return windowSlideMinutes; }
    public void setWindowSlideMinutes(int windowSlideMinutes) { this.windowSlideMinutes = windowSlideMinutes; }

    public String getCheckpointDir() { return checkpointDir; }
    public void setCheckpointDir(String checkpointDir) { this.checkpointDir = checkpointDir; }

    public int getCheckpointIntervalMs() { return checkpointIntervalMs; }
    public void setCheckpointIntervalMs(int checkpointIntervalMs) { this.checkpointIntervalMs = checkpointIntervalMs; }

    public int getParallelism() { return parallelism; }
    public void setParallelism(int parallelism) { this.parallelism = parallelism; }

    public int getMaxOutOfOrdernessSeconds() { return maxOutOfOrdernessSeconds; }
    public void setMaxOutOfOrdernessSeconds(int maxOutOfOrdernessSeconds) { this.maxOutOfOrdernessSeconds = maxOutOfOrdernessSeconds; }

    public double getCtrPriorClicks() { return ctrPriorClicks; }
    public void setCtrPriorClicks(double ctrPriorClicks) { this.ctrPriorClicks = ctrPriorClicks; }

    public double getCtrPriorViews() { return ctrPriorViews; }
    public void setCtrPriorViews(double ctrPriorViews) { this.ctrPriorViews = ctrPriorViews; }

    public Duration getWindowSize() { return Duration.ofHours(windowSizeHours); }
    public Duration getWindowSlide() { return Duration.ofMinutes(windowSlideMinutes); }
    public Duration getMaxOutOfOrderness() { return Duration.ofSeconds(maxOutOfOrdernessSeconds); }

    private String redisPassword = System.getenv().getOrDefault("REDIS_PASSWORD", "");
    public String getRedisPassword() { return redisPassword; }
}