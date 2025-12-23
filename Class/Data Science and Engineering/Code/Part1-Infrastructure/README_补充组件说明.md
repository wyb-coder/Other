# 补充组件使用说明

## 1. Flume 数据采集

### 配置文件位置

- `Code/Part1-Infrastructure/flume-traffic.conf`
- `Code/Part3-Storage/flume-traffic.conf`

### 架构设计

```
CSV 文件 → Spooldir Source → File Channel → [HDFS Sink + HBase Sink]
                                              ↓            ↓
                                         Batch Layer   Speed Layer
```

### 启动 Flume

```bash
# 1. 启动完整集群（包含 Flume）
docker-compose up -d

# 2. 放置 CSV 文件到监控目录
docker cp Data/82_processed.csv flume:/data/incoming/

# 3. 查看 Flume 日志
docker logs -f flume
```

## 2. Spark 分布式计算

### 容器配置

- **Spark Master**: http://localhost:8080
- **Spark Worker**: http://localhost:8081

### 提交 Spark 作业

```bash
# 1. 将脚本复制到 Spark 容器
docker cp Code/Part5-Aggregation/spark_aggregation_job.py spark-master:/opt/spark-apps/

# 2. 提交作业
docker exec spark-master spark-submit \
    --master spark://spark-master:7077 \
    /opt/spark-apps/spark_aggregation_job.py

# 3. 查看作业状态
# 访问 http://localhost:8080 查看 Spark UI
```

### 本地 PySpark 开发

```bash
# 安装 PySpark
pip install pyspark

# 本地运行
python Code/Part5-Aggregation/spark_aggregation.py
```

## 3. 完整启动顺序

```bash
cd Code/Part1-Infrastructure

# 启动所有服务
docker-compose up -d

# 检查服务状态
docker-compose ps

# 服务访问地址：
# - HDFS NameNode: http://localhost:50070
# - YARN: http://localhost:8088
# - HBase Master: http://localhost:16010
# - Spark Master: http://localhost:8080
```

## 4. 服务依赖图

```
                    ┌─────────────┐
                    │  ZooKeeper  │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
    │NameNode │       │HBase    │       │Spark    │
    │DataNode │       │Master   │       │Master   │
    └────┬────┘       │Region   │       │Worker   │
         │            └────┬────┘       └────┬────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                    ┌──────▼──────┐
                    │    Flume    │
                    │   Agent     │
                    └─────────────┘
```
