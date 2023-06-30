---
layout:     post
title:      "kubernetes 基础学习"
date:       2019-12-12 19:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - Kubernetes
---



keypoint：

- 基础概念：Pod，控制器类型，网络通讯模式；
- 安装：构建K8S集群；
- 资源清单：资源、资源清单语法，编写Pod、Pod生命周期；
- Pod控制器：什么是控制器，控制器类型、特点、使用定义方式；
- 服务发现：Service原理及其构建方式；
- 存储：configMap、Secret、volume、PV，多种存储的特点，不同环境选择合适方案；
- 调度器：原理、把Pod定义到指定节点运行；
- 集群安全：认证、鉴权、准入控制 原理及其流程；
- HELM：相当于Linux的yum，helm原理、模板自定义、部署常用插件；
- 运维：源码修改、证书可用期限，高可用构建；

## 组件

![1574241175925](/img/K8S/k8s-架构.png)

主要组件：

- APIServer：提供了资源操作的唯一入口，并提供认证、授权、访问控制、API 注册和发现等机制；
- CrontrollerManager：负责维护集群的状态，比如故障检测、自动扩展、滚动更新等；
- Scheduler：负责资源的调度，按照预定的调度策略将 Pod 调度到相应的机器上；
- etcd：键值对数据库，储存K8S集群所有重要信息（持久化）；
- Kubelet：负责维持容器的生命周期，同时也负责 Volume（CVI）和网络（CNI）的管理；
- Kube-proxy：负责为 Service 提供 cluster 内部的服务发现和负载均衡，写入规则至 IPTABLES、IPVS 实现服务映射访问的；

其他组件：

- COREDNS：可以为集群中的SVC创建一个域名IP的对应关系解析；
- DASHBOARD：给 K8S 集群提供一个 B/S 结构访问体系；
- INGRESS CONTROLLER：官方只能实现四层代理，INGRESS 可以实现七层代理；
- FEDERATION：提供一个可以跨集群中心多K8S统一管理功能；
- PROMETHEUS：提供K8S集群的监控能力；
- ELK：提供 K8S 集群日志统一分析介入平台；

## K8s资源

K8s中所有的内容都抽象为资源，资源实例化之后，叫做对象。

### 分类

名称空间级别：

- 工作负载型资源（workload）：Pod、ReplicaSet、Deployment、StatefulSet、DaemonSet、Job、CronJob（ReplicationController在v1.11版本被废弃）
- 服务发现及负载均衡型资源（ServiceDiscovery LoadBalance）：Service、Ingress、...
- 配置与存储型资源：Volume、CSI（容器存储接口，可以扩展各种各样的第三方存储卷）
- 特殊类型的存储卷：ConfigMap（当配置中心来使用的资源类型）、Secret（保存敏感数据）、DownwardAPI（把外部环境中的信息输出给容器

集群级资源：

- Namespace、Node、Role、ClusterRole、RoleBinding、ClusterRoleBinding

元数据类型资源：

- HPA、PodTemplate、LimitRange

### 资源清单

用yaml格式的文件来创建符合预期期望的pod，这样的yaml文件称为资源清单。

分为两部分：

- 控制器定义
- 被控制对象

字段含义：

| 字段       | 含义       |
| ---------- | ---------- |
| apiVersion | API版本    |
| kind       | 资源类型   |
| metadata   | 资源元数据 |
| spec       | 资源规格   |
| replicas   | 副本数量   |
| selector   | 标签选择器 |
| template   | Pod模板    |
| metadata   | Pod元数据  |
| spec       | Pod规格    |
| containers | 容器配置   |

生成yaml文件模板：

```shell
kubectl create deployment web --image=nginx -o yaml --dry-run > my1.yaml
```

## Pod

### Pod简介

- 最小部署的单元，其他的资源对象都是用来支撑或者扩展Pod对象功能的（比如：控制器、Service/Ingress、PersistentVolume）；
- 包含多个容器（一组容器的集合，Pause根容器 + 一个或多个业务容器）；
- 一个Pod中容器共享网络命名空间；
- Pod是短暂的；

#### Pod存在的意义

（1）创建容器使用docker，一个docker对应一个容器，一个容器有进程，docker设计为一个容器运行一个应用程序；

（2）Pod是多进程设计，一个Pod里面有多个容器，运行多个应用程序；

（3）Pod存在为了亲密性应用：

- 两个应用之间进行交互；
- 网络之间调用：socket 、127.0.0.1；
- 两个应用需要频繁调用

#### Pod实现机制

**（1）共享网络**

- docker容器：容器本身之间相互隔离（namespace、group）；
- 共享网络前提条件：容器在同一个namespace；

Pod实现共享网络机制：

- Pause根容器（info容器），独立ip、mac、port；
- 业务容器：加入info容器，同一个namespace中；

**（2）共享存储**

- pod持久化数据：日志数据、业务数据；
- 持久化存储：volume存储卷；

#### Pod镜像拉取策略

imagePullPolicy：

- IfNotPresent：默认值，不存在则拉取；
- Always：每次创建Pod重新拉取；
- Never：不会主动拉取。

#### Pod重启策略

restartPolicy：

- Alaways：默认，总是重启；
- OnFailure：异常退出才重启；
- Nerver：当容器终止退出，从不重启。

#### Pod资源限制

```yaml
spec:
  containers:
  - name: db
  	image: mysql
  	env:
  	- name: MYSQL_ROOT_PASSWORD
  	  value: "password"
  	
  	# 资源限制
    resources:
      # 最小需求限制
      requests:
        memory: "64Mi"
        cpu: "250m"
      # 最大使用限制
      limits:
        memory: "128Mi"
        cpu: "500m"
```

说明：

1. 当集群中的计算资源不很充足, 如果集群中的pod负载突然加大, 就会使某个node的资源严重不足, 为了避免系统挂掉, 该node会选择清理某些pod来释放资源, 此时每个pod都可能成为牺牲品。

2. kubernetes保障机制：

- 限制pod进行资源限额
- 允许集群资源被超额分配, 以提高集群的资源利用率
- 为pod划分等级, 确保不同的pod有不同的服务质量qos, 资源不足时， 低等级的pod会被清理, 确保高等级的pod正常运行

3. kubernetes会根据Request的值去查找有足够资源的node来调度此pod

- limit对应资源量的上限, 既最多允许使用这个上限的资源量, 由于cpu是可压缩的, 进程是无法突破上限的, 而memory是不可压缩资源, 当进程试图请求超过limit限制时的memory, 此进程就会被kubernetes杀掉
- 对于cpu和内存而言, **pod的request和limit是指该pod中所有容器的 Requests或Limits的总和**,
  - 例如： 某个节点cpu资源充足, 而内存为4G，其中3GB可以运行pod, 而某个pod的memory request为1GB, limit为2GB， 那么这个节点上最多可以运行3个这样的pod
- **待调度pod的request值总和超过该节点提供的空闲资源, 不会调度到该节点node上。**

#### Pod健康检查

- livenessProbe（**存活检查**）：如果检查失败，将杀死容器，根据重启策略操作；
- readinessProbe（**就绪检查**）：如果检查失败，会把pod从service endpoints中剔除；

Probe支持三种检查方法：

- httpGet：发送HTTP请求，返回200-400范围状态码为成功；
- exec：执行shell命令返回状态码是0为成功；
- tcpSocket：发起TCP socket建立成功。

示例：

```yaml
spec:
  containers:
  - name: liveness
  	image: busybox
  	args:
  	- /bin/sh
  	- -c
  	- touch /tmp/healthy; sleep 30; rm -rf /tmp/healthy
  	# 存活检查
  	livenessProbe:
  	  exec:
  	  	command:
  	  	- cat
  	  	- /tmp/healthy
  	  initialDelaySeconds: 5
  	  periodSeconds: 5
```

### Pod工作流

- [Kubernetes Pod 工作流](https://www.qikqiak.com/post/pod-workflow/)

`apiserver`是整个集群的控制入口，`etcd`在集群中充当数据库的作用，只有`apiserver`才可以直接去操作`etcd`集群，而我们的`apiserver`无论是对内还是对外都提供了统一的`REST API`服务。组件之间当然也是通过`apiserver`进行通信的，其中`kube-controller-manager`、`kube-scheduler`、`kubelet`是通过`apiserver watch API`来**监控**我们的资源变化，并且对资源的相关状态更新操作也都是通过`apiserver`进行的，所以说白了组件之间的通信就是通过`apiserver REST API`和`apiserver watch API`进行的。

Pod工作流：

![workflow](/img/K8S/pod-workflow.png)

master节点：

- 第一步通过`apiserver REST API`创建一个`Pod`；
- 然后`apiserver`接收到数据后将数据写入到`etcd`中；
- 由于`kube-scheduler`通过`apiserver watch API`一直在**监听**资源的变化，这个时候发现有一个新的`Pod`，但是这个时候该`Pod`还没和任何`Node`节点进行绑定，所以`kube-scheduler`就经过一系列复杂的调度策略，选择出一个合适的`Node`节点，将该`Pod`和该目标`Node`进行绑定，当然也会更新到`etcd`中去的；

node节点：

- 这个时候一样的目标`Node`节点上的`kubelet`通过`apiserver watch API`**检测**到有一个新的`Pod`被调度过来了，他就将该`Pod`的相关数据传递给后面的容器运行时(`container runtime`)，比如`Docker`，让他们去运行该`Pod`；
- 而且`kubelet`还会通过`container runtime`获取`Pod`的状态，然后更新到`apiserver`中，当然最后也是写入到`etcd`中去的。

#### 节点调度

影响因素：

- **资源限制**：根据resources-requests找到足够资源的node节点进行调度；
- **nodeSelector（节点选择器**）：会弃用；
- **nodeAffinity（节点亲和性**）：也是根据节点标签选择
  - 硬亲和性：约束条件必须满足；
  - 软亲和性：尝试满足，不保证；
  - operator操作符：In、NotIn、Exists、DoesNotExists、Gt（大于）、Lt（小于）；
- **污点和污点容忍**：
  - Taint污点：节点不做普通分配调度，是节点属性；
  - 应用场景：专用节点、配置特定硬件节点、基于Taint驱逐；
  - 污点值有三个：
    - NoSchedule（一定不被调度）
    - PreferNoSchedule（尽量不被调度）
    - NoExecute（不会调度，并且还会驱逐Node已有Pod）。

### Pod生命周期

[Kubernetes Pod 生命周期](http://docs.kubernetes.org.cn/719.html)

**phase**

Pod 的 status 定义在 PodStatus对象中，其中有一个 phase 字段。

Pod 的相位（phase）是 Pod 在其生命周期中的简单宏观概述。该阶段并不是对容器或 Pod 的综合汇总，也不是为了做为综合状态机。

phase 可能的值：

- 挂起（Pending）：Pod 已被 Kubernetes 系统接受，但有一个或者多个容器镜像尚未创建。等待时间包括调度 Pod 的时间和通过网络下载镜像的时间，这可能需要花点时间。
- 运行中（Running）：该 Pod 已经绑定到了一个节点上，Pod 中所有的容器都已被创建。至少有一个容器正在运行，或者正处于启动或重启状态。
- 成功（Succeeded）：Pod 中的所有容器都被成功终止，并且不会再重启。
- 失败（Failed）：Pod 中的所有容器都已终止了，并且至少有一个容器是因为失败终止。也就是说，容器以非0状态退出或者被系统终止。
- 未知（Unknown）：因为某些原因无法取得 Pod 的状态，通常是因为与 Pod 所在主机通信失败。

**Init容器**

Pod能够有多个容器，应用运行在容器里面，但是它也可能有一个或多个先于应用容器启动的Init容器。

Init容器与普通的容器非常像，除了如下两点：

- Init容器总是运行到成功完成为止
- 每个Init容器都必须在下一个Init容器启动之前成功完成

如果Pod的Init容器失败，Kubernetes会不断地重启该Pod，值到Init容器成功为止。然而，如果Pod对应地restartPolicy为Never，它不会重新启动。

**探针**

探针是由 kubelet对容器执行的定期诊断。

每次探测都将获得以下三种结果之一：

- 成功：容器通过了诊断。
- 失败：容器未通过诊断。
- 未知：诊断失败，因此不会采取任何行动。

Kubelet 可以选择是否执行在容器上运行的两种探针执行和做出反应：

- `livenessProbe`：指示容器是否正在运行。如果存活探测失败，则 kubelet 会杀死容器，并且容器将受到其重启策略的影响。如果容器不提供存活探针，则默认状态为 `Success`。
- `readinessProbe`：指示容器是否准备好服务请求。如果就绪探测失败，端点控制器将从与 Pod 匹配的所有 Service 的端点中删除该 Pod 的 IP 地址。初始延迟之前的就绪状态默认为 `Failure`。如果容器不提供就绪探针，则默认状态为 `Success`。

## Controller

**Pod和Controller关系：**

- Pod是通过Controller实现应用的运维，比如伸缩、滚动升级等；
- Pod和Controller之间通过label标签建立关系。

**无状态和有状态Controller：**

- 无状态：**Deployment**
  - 认为Pod都是一样的；
  - 没有顺序要求；
  - 不用考虑在哪个node运行；
  - 随意进行伸缩和扩展

- 有状态：**StateFulSet**
  - 上面无状态因素都需要考虑到；
  - 让每个pod独立，保持pod启动顺序和唯一性
    - 唯一的网络标识符，持久存储；
    - 有序，比如MySQL主从；

资源控制器：类型、特点、使用场景。

### ReplicationController和ReplicaSet

ReplicationController（RC）用来确保容器应用地副本数始终保持在用户定义地副本数，即如果有容器异常退出，会自动创建新地Pod来替代，如果异常多出来地容器也会自动回收；

在新版中建议使用ReplicaSet（RS）来取代RC。ReplicaSet跟ReplicationController没有本质地不同，只是名字不一样，并且ReplicaSet支持集合式地selector。

### Deployment

Deployment为Pod和ReplicaSet提供了一个声明式（declarative）方法，用来替代以前地ReplicationController来方便地管理应用。

- 通过RS控制Pod
- 滚动更新、回滚
- 扩容和缩容
- 暂停和继续Deployment

### StateFulSet

StatefulSet作为Controller为Pod提供**唯一的标识**，它可以保证部署和scale的**顺序**。

StatefulSet是为了解决有状态服务的问题（对应Deployment和ReplicaSet是为无状态服务而设计），其应用场景包括：

- **稳定的持久化存储**，即Pod重新调度后还是能访问到相同的持久化数据，基于PVC来实现；
- **稳定的网络标志**，即Pod重新调度后其PodName和HostName不变，基于HeadlessService（即没有Cluster IP的Service）来实现；
- **有序部署，有序扩展**，即Pod是有顺序的，在部署或扩展的时候要依据定义的顺序依次执行（0到N-1），基于Init container来实现；
- **有序收缩，有序删除**（N-1到0）。

StateFulSet和Deployment的区别：有身份的（唯一标识）

- 根据主机名 + 按照一定规则生成域名
- 格式：主机名称.service名称.名称空间.svc.cluster.local。

### DaemonSet

DaemonSet确保全部（或者一些）Node上运行一个Pod地副本。当有Node加入集群时，也会为他们新增一个Pod；当有Node从集群移除时，这些Pod也会被回收；删除DaemonSet将会删除它创建地所有Pod。

- **运行集群存储daemon**，例如在每个Node上运行glusterd、ceph；
- **在每个Node上运行日志收集daemon**，例如fluentd、logstash；
- **在每个Node上运行监控daemon**，例如PrometheusNodeExporter、collectd、Datadog代理；

### Job/CronJob

Job负责批处理任务，即**仅执行一次地任务**，它保证批处理任务地一个或多个Pod成功结束。

CronJob管理基于时间的Job，**定时任务**，即在给定时间点运行一次、周期性地在给定时间点运行（数据库备份、发送邮件）。

### Horizontal Pod Autoscaling

应用资源使用率通常都有高峰和低估的时候，削峰填谷，提高集群的整体资源利用率，让service中的Pod个数自动调整，依赖于HPA，顾名思义，使Pod水平自动缩放。

## 服务、负载均衡和联网

Kubernetes 网络解决四方面的问题：

- 一个 Pod 中的容器之间通过本地回路（loopback）通信。
- 集群网络在不同 pod 之间提供通信。
- Service 资源允许你对外暴露 Pods 中运行的应用程序，以支持来自于集群外部的访问。
- 可以使用 Services 来发布仅供集群内部使用的服务。

### Service

[服务](https://kubernetes.io/zh/docs/concepts/services-networking/service/)	[Service](https://kubernetes.feisky.xyz/concepts/objects/service)

Service定义了一种抽象：一个Pod的逻辑分组，一种可以访问它们的策略（微服务）。这一组Pod能够被Service访问到，通常是通过Label Selector。

Service存在的意义：

- 服务发现
- 负载均衡

Pod和Service关系：

- 根据label标签和selector建立关联。

类型：

- **ClusterIP**：默认类型，自动分配一个仅Cluster内部可以访问的虚拟IP；
- **NodePort**：在ClusterIP基础上为Service在每台机器上绑定一个端口，这样就可以通过:NodePort来访问该服务；
- **LoadBalancer**：在NodePort的基础上，借助cloud provider创建一个外部负载均衡器，并将请求转发到NodePort；
- **ExternalName**：把集群外部的服务引入到集群内部来，在集群内部直接使用，没有任何类型代理被创建（只有1.7或更高版本的kube-dns才支持）。

#### Headless Services

- [headless-services](https://kubernetes.io/zh/docs/concepts/services-networking/service/#headless-services)

无头服务（Headless Services）:

- clusterIP: none

有时不需要或不想要负载均衡，以及单独的 Service IP。 遇到这种情况，可以通过指定 Cluster IP（`spec.clusterIP`）的值为 `"None"` 来创建 `Headless` Service。

你可以使用无头 Service 与其他服务发现机制进行接口，而不必与 Kubernetes 的实现捆绑在一起。

对这无头 Service 并不会分配 Cluster IP，kube-proxy 不会处理它们， 而且平台也不会为它们进行负载均衡和路由。 DNS 如何实现自动配置，依赖于 Service 是否定义了选择算符。

### Ingress

[ingress](https://kubernetes.io/zh/docs/concepts/services-networking/ingress/)

- Ingress 是对集群中服务的外部访问进行管理的 API 对象，典型的访问方式是 HTTP。
- Ingress 可以提供负载均衡、SSL 终结和基于名称的虚拟托管。

**Ingress**公开了从集群外部到集群内**服务**的 HTTP 和 HTTPS 路由。 流量路由由 Ingress 资源上定义的规则控制。

**Ingress和Pod的关系：**

- pod和ingress通过service关联；
- ingress作为统一入口，由service关联一组pod；

```text
ingress:
  	a.com -> service a -> pod a-1、pod a-2...
  	b.com -> service b -> pod b-1、pod b-2...
```

**使用：**

- Service使用NodePort对外暴露端口；
- 部署 Ingress Controller（官方nginx控制器，ingress-controller.yaml）；
- 创建 Ingress 规则。

```yaml
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: example-ingress
spec:
  rules:
  - host: example.ingressdemo.com
    http:
      paths:
      - path: /
        backend:
          serviceName: web
          servicePort: 80
```

### DNS

- [dns-pod-service](https://kubernetes.io/zh/docs/concepts/services-networking/dns-pod-service/)

Kubernetes DNS 在群集上调度 DNS Pod 和服务，并配置 kubelet 以告知各个容器使用 DNS 服务的 IP 来解析 DNS 名称。

## 配置

###  Secret

- 作用：加密数据存在etcd里面，让Pod容器以变量或挂载Volume方式进行访问；
- 场景：凭证；

示例：

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mysecret
type: Opaque
data:
  # base64编码
  username: YWRtaW4=
  password: MWYyZDFlMmU2N2Rm
```

以变量形式挂载到Pod容器中：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myPod
spec:
  containers:
  - name: nginx
    image: nginx
    env:
      # 变量挂载
      - name: SECRET_USERNAME
        valueFrom:
          secretKeyRef:
            name: mysecret
            key: username
      - name: SECRET_PASSWORD
        valueFrom:
          secretKeyRef:
            name: mysecret
            key: password
```

以Volume形式挂载到Pod容器中：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myPod
spec:
  containers:
  - name: nginx
    image: nginx
    # Volume挂载
    volumeMounts: 
    - name: foo
      mountPath: "/etc/foo"
      readOnly: true
  # volumes
  volumes:
  - name: foo
    secret:
      secretName: mysecret
```

###  Configmap

- 作用：存储不加密数据到etcd，让Pod以变量或者Volume挂载到容器中，同Secret；
- 场景：配置文件；

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myconfig
  namespace: default
data:
  log.level: info
```

## 存储

###  Volume

NFS 网络存储：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-nfs
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        # 挂载nfs
        volumeMounts:
        - name: wwwroot
          mountPath: /usr/share/nginx/html
        ports:
        - containerPort: 80
      volumes:
        - name: wwwroot
          # nfs volume
          nfs:
          	server: 192.168.44.134
          	path: /data/nfs
```

###  PV和PVC

- PV：持久卷，持久化存储，对存储资源进行抽象，对外提供可以调用的地方（生产者）；
- PVC：持久卷申请，用于调用，不需要关心内部实现细节（消费者）。

应用部署 --> PVC --> PV。

示例：

```yaml
# 应用
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-nfs
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        volumeMounts:
        - name: wwwroot
          mountPath: /usr/share/nginx/html
        ports:
        - containerPort: 80
      volumes:
        - name: wwwroot
          # PVC
          persistentVolumeClaim:
            claimName: my-pvc
---
# PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
      
---
# PV
apiVersion: v1
kind: PersistentVolume
metadata:
  name: my-pv
spec:
  capacity: 
    storage: 5Gi
  accessModes:
    - ReadWriteMany
  nfs:
    path: /k8s/nfs
    server: 192.168.44.134 
```

## 集群安全机制

- [Kubernetes API 访问控制](https://kubernetes.io/zh/docs/concepts/security/controlling-access/)

访问k8s集群的时候，需要经过三个步骤：

- 认证
- 鉴权
- 准入控制

进行访问的时候，都需要经过apiServer：

- apiServer做统一协调；
- 访问过程中需要证书、token、或者用户名+密码；
- 如果访问Pod，需要ServiceAccount。

###  认证

传输安全：对外不暴露8080端口，只能内部访问，对外使用端口6443；

客户端身份认证常用方式：

- **https 证书认证**：基于ca证书，kube-apiserver 、etcd、kubelet 连接kube-apiserver、kube-proxy连接 kube-apiserver 均采用 https传输方式；
- **http token认证**：通过token识别用户；
- **http 基本认证**：用户名+密码认证，在k8s中基本很少使用。

###  鉴权（授权）

授权模式：

- Node
- Webhook
- ABAC：基于属性的访问控制；
- RBAC：基于角色的访问控制。

基于 **RBAC（Role-Based Access Control，基于角色的访问控制）** 进行鉴权操作。

- **角色** 
  - Role：授权特定命名空间的访问权限
  - ClusterRole：授权所有命名空间的访问权限
- **主体（subject）** 
  - User：用户
  - Group：用户组
  - ServiceAccount：服务账号
- **角色绑定**
  - RoleBinding：将角色绑定到主体（即subject）
  - ClusterRoleBinding：将集群角色绑定到主体

角色访问规划：

- API：请求路径、请求方法（get、list、create、update、patch、watch、delete）；
- http请求方法：get、post、put、delete；
- 资源：pod、node...
- 子资源、命名空间、API组...

### 准入控制

Adminssion Control实际上是一个准入控制器插件列表，发送到API Server的请求都需要经过这个列表中的每个准入控制器 插件的检查，检查不通过，则拒绝请求。

## Helm

部署应用：Deployment、Service、Ingress。

缺点：部署微服务，可能有几十个服务，每个服务都有一套yaml文件，需要维护大量yaml文件，版本管理特别不方便。

使用helm：

- 可以把这些微服务的yaml作为一个整体管理；
- 实现yaml高效复用；
- 实现应用级别的版本管理；

Helm是一个Kubernetes 的包管理工具，就像Linux下的包管理器，如yum/apt等，可以很方便的将之前打包好的yaml文件部署到kubernetes上。

#### 三个重要概念

- **helm**：是一个命令行客户端工具；
- **Chart**：应用描述，一系列用于描述k8s资源相关文件的集合；
- **Release**：基于Chart的部署实体，一个chart被helm运行后将会生成对应的一个release；将在k8s中创建出真实运行的资源对象。

## 监控

### 监控指标

- 集群监控：
  - 节点资源利用率；
  - 节点数；
  - 运行pods；
- Pod监控：
  - 容器指标：CPU、内存；
  - 应用程序状态；

### 监控平台

组件：Prometheus + Grafana，Prometheus定时抓取数据，Grafana进行展示。

## 高可用

- [利用 kubeadm 创建高可用集群](https://kubernetes.io/zh/docs/setup/production-environment/tools/kubeadm/high-availability/)
- [搭建高可用的 Kubernetes Masters](https://kubernetes.io/zh/docs/tasks/administer-cluster/highly-available-master/)

## API-Fabric
- [fabric8io/kubernetes-client](https://github.com/fabric8io/kubernetes-client)

- [Fabric8浅析](https://www.jianshu.com/p/38ff7d64cfde)
