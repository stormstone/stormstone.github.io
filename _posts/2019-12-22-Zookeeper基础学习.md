---
layout:     post
title:      "Zookeeper 基础学习"
date:       2019-12-22 19:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - Zookeeper
    - BBigData
---

# Zookeeper

- [https://zookeeper.apache.org/](https://zookeeper.apache.org/)
- [https://github.com/apache/zookeeper](https://github.com/apache/zookeeper)


zk核心：
- [ZooKeeper Programmer's Guide](https://zookeeper.apache.org/doc/current/zookeeperProgrammers.html)
  - 数据模型
  - Session会话
  - Watches监听
  - ACL访问控制
  - 一致性保证

用法：
- [zk进阶--集群--分布式管理--注册中心--分布式JOB--分布式锁](https://github.com/qiurunze123/zookeeperDesign/blob/master/docs/zkprocess2.md)
  - 分布式集群管理
  - 分布式注册中心
  - 分布式锁


## 一、简介

> ZooKeeper is a distributed, open-source coordination service for distributed applications. 
> It exposes a simple set of primitives that distributed applications can build upon to implement higher level services for synchronization, configuration maintenance, and groups and naming. 
> It is designed to be easy to program，and uses a data model styled after the familiar directory tree structure of file systems. 
> It runs in Java and has bindings for both Java and C.

ZooKeeper 就是一个**分布式**的，**开放源码**的分布式应用程序**协调服务**。它使得应用开发人员可以更多的关注应用本身的逻辑，而不是协同工作上。
从系统设计看，ZooKeeper 从文件系统 API 得到启发，提供一组简单的 API，使得开发人员可以实现通用的协作任务，例如选举主节点，管理组内成员的关系，管理元数据等，同时 ZooKeeper 的服务组件运行在一组专用的服务器之上，也保证了高容错性和可扩展性。

分布式系统中关键在于**进程通信**，其有两种选择：直接通过网络进行信息交换，或者读写某些共享存储。
对于 **ZooKeeper 实现协作和同步原语本质上是使用共享存储模型**，即开发的应用是连接到 ZooKeeper 服务器端的客户端，他们连接到 ZooKeeper 服务器端进行相关的操作，以来影响服务器端存储的共享数据，最终应用间实现协作。

## 二、架构

Zookeeper从设计模式角度来理解：是一个基于**观察者设计模式**设计的分布式服务管理框架，它负责存储和管理大家都关心的数据，然后接受观察者的注册，一旦这些数据的状态发生变化，zookeeper就将负责通知已经在zookeeper上注册的那些观察者做出相应的反应。

**zookeeper = 文件系统 + 通知机制。**（协调服务）

### ZNode

Zookeeper的核心是一个精简的文件系统，它提供一些简单的操作和一些额外的抽象操作，例如，排序和通知。

![img](/img/Zookeeper/ZooKeeper数据树结构.png)

zookeeper数据模型的结构与Unix文件系统很类似，整体上可以看作是一棵树，每个节点称作一个ZNode（没有目录、文件区别）。每一个ZNode默认能够存储1MB的数据，每个ZNode都可以通过其路径唯一标识。

### 节点类型

- **有序、无序**：序号是全局顺序编号；
- **临时、永久**：客户端和服务器端断开连接后，节点是否删除；

四种节点类型：

1. **持久化目录节点**：客户端与zookeeper断开连接后，该节点依旧存在；
2. **持久化顺序编号目录节点**：客户端与zookeeper断开连接后，该节点依旧存在，zookeeper给该节点进行顺序编号；
3. **临时目录节点**：客户端与zookeeper断开连接后，该节点被删除；
4. **临时顺序编号目录节点**：客户端与zookeeper断开连接后，该节点被删除，zookeeper给该节点进行顺序编号。

### 节点状态

Stat结构体：

- **czxid**：创建节点的事务zxid，每次修改ZooKeeper状态都会收到一个zxid形式的时间戳，也就是ZooKeeper事务ID，事务ID是ZooKeeper中所有修改总的次序，每个修改都有唯一的zxid，如果zxid1小于zxid2，那么zxid1在zxid2之前发生；
- ctime：znode被创建的毫秒数(从1970年开始)；
- **mzxid**：znode最后更新的事务zxid；
- mtime：znode最后修改的毫秒数(从1970年开始)；
- **pZxid**：znode最后更新的子节点zxid；
- cversion：znode子节点变化号，znode子节点修改次数；
- dataversion：znode数据变化号；
- aclVersion：znode访问控制列表的变化号；
- ephemeralOwner：如果是临时节点，这个是znode拥有者的session id，如果不是临时节点则是0；
- dataLength：znode的数据长度；
- numChildren：znode子节点数量。

### 服务器和客户端

ZooKeeper服务器和客户端架构：

![ZooKeeper Service](/img/Zookeeper/ZooKeeper服务器和客户端架构.jpg)

ZooKeeper服务器和客户端工作流程：

![ZooKeeper Components](/img/Zookeeper/ZooKeeper服务器和客户端工作流程.jpg)

ZooKeeper 客户端在服务器集群中执行任何请求前必须先与服务器建立会话（**session）**，客户端提交给 ZooKeeper 的所有操作均关联在一个会话上。客户端初始化连接到集合中某个服务器或一个独立的服务器，客户端提供**TCP 协议**与服务器进行连接并通信，但当会话无法与当前连接的服务器继续通信时，会话就可能转移到另外一个服务器，ZooKeeper 客户端**透明地**转移一个会话到不同的服务器。需要指明的，会话提供了顺序保障，同一个会话中的请求会以 FIFO（先进先出）顺序执行。

#### 监听器原理

```reStructuredText
-- new Zookeeper()
    -- new ClientCnxn()
    	-- new SendThread()
    	-- new EventThread()
```

- 在main线程中创建Zookeeper客户端，这时就会创建两个线程，一个负责网络连接通信（connect），一个负责监听（listener）；
- 通过connect线程将注册的监听事件发送给zookeeper；
- 在zookeeper的注册监听器列表中将注册的监听事件添加到列表中；
- zookeeper监听到有数据或路径变化，就会将这个消息发送给listener线程；
- listener线程内部调用process()方法

![1577019725337](/img/Zookeeper/zookeeper_监听器原理.png)

常见的监听：

- 监听节点数据的变化，get path watch；
- 监听子节点增减的变化，ls path watch。

### 特点

- 1）一个领导者（Leader），多个跟随者（Follower）组成的集群；
- 2）集群中只要有半数以上节点存活，集群就能正常服务；
- 3）全局数据一致：每个Server保存一份相同的数据副本，Client无论连接到哪个Server，数据都是一致的；
- 4）更新请求顺序进行，来自同一个Client的更新请求按其发送顺序依次执行；
- 5）数据更新原子性，一次数据更新要么成功，要么失败；
- 6）实时性，在一定时间范围内，Client能读到最新数据。

### 客户端命令

| 命令基本语法       | 功能描述                                               |
| ------------------ | ------------------------------------------------------ |
| help               | 显示所有操作命令                                       |
| ls path [watch]    | 使用 ls 命令来查看当前znode中所包含的内容              |
| ls2 path   [watch] | 查看当前节点数据并能看到更新次数等数据                 |
| create             | 普通创建   -s  含有序列   -e  临时（重启或者超时消失） |
| get path   [watch] | 获得节点的值                                           |
| set                | 设置节点的具体值                                       |
| stat               | 查看节点状态                                           |
| delete             | 删除节点                                               |
| rmr                | 递归删除节点                                           |

### 应用场景

HA（High Available）、Kafka、Hbase；统一命名服务、统一配置管理、统一集群管理、软负载均衡。

**Apache Kafka**

Kafka 是一个基于发布-订阅模型的消息系统。其中 ZooKeeper 用于检测崩溃，实现主题的发现，并保持主题的生产和消费状态。

**Apache HBase**

HBase 是一个通常与 Hadoop 一起使用的数据存储仓库。在 HBase 中，ZooKeeper 用于选举一个集群内的主节点，以便跟踪可用的服务器，并保持集群的元数据。

**Apache Solr**

Solr 是一个企业级的搜索平台，它使用 ZooKeeper 来存储集群的元数据，并协作更新这些元数据。

## 三、内部原理

总体来说 ZooKeeper 运行于一个集群环境中，选举出某个服务器作为**群首（Leader）**，其他服务器追随群首（**Follower**）。群首作为中心处理所有对 ZooKeeper 系统变更的请求，它就像一个定序器，建立了所有对 ZooKeeper 状态的更新的顺序，追随者接收群首所发出更新操作请求，并对这些请求进行处理，以此来保障状态更新操作不会发生碰撞。

### 群首选举

群首为集群中的服务器选择出来的一个服务器，并会一直被集群所认可。设置群首的目的是为了对客户端所发起的 ZooKeeper 状态更新请求进行**排序**，包括 create，setData 和 delete 操作。群首将每一个请求转换为一个事务，将这些事务发送给追随者，**确保集群按照群首确定的顺序接受并处理这些事务**。

每个服务器启动后进入 **LOOKING** 状态，开始选举一个新的群首或者查找已经存在的群首。如果群首已经存在，其他服务器就会通知这个新启动的服务器，告知哪个服务器是群首，于此同时，新服务器会与群首建立连接，以确保自己的状态与群首一致。

如果群所有的服务器均处于 LOOKING 状态，这些服务器之间就会进行通信来选举一个群首，通过信息交换对群首选举达成共识的选择。在本次选举过程中胜出的服务器将进入 **LEADING** 状态，而集群中其他服务器将会进入 **FOLLOWING** 状态。

具体看，一个服务器进入 LOOKING 状态，就会发送向集群中每个服务器发送一个通知信息，该消息中包括该服务器的**投票（vote）信息**，投票中包含**服务器标识符（sid）**和**最近执行事务的 zxid** 信息。

当一个服务器收到一个投票信息，该服务器将会根据以下规则修改自己的投票信息：

1. 将接收的 voteId 和 voteZxid 作为一个标识符，并获取接收方当前的投票中的 zxid，用 myZxid 和 mySid 表示接收方服务器自己的值。
2. 如果（voteZxid > myZxid）或者（voteZxid == myZxid 且 voteId >mySid）,保留当前的投票信息。
3. 否则，修改自己的投票信息，将 voteZxid 赋值给 myZxid，将 voteId 赋值给 mySid。

从上面的投票过程可以看出，只有最新的服务器将赢得选举，因为其拥有最近一次的 zxid。如果多个服务器拥有的最新的 zxid 值，其中的 sid 值最大的将会赢得选举。

当一个服务器连接到仲裁数量的服务器发来的投票都一样时，就表示群首选举成功，如果被选举的群首为某个服务器自己，该服务器将会开始行使群首角色，否则就会成为一个追随者并尝试连接被选举的群首服务器。一旦连接成功，追随者和群首之间将会进行**状态同步**，在同步完成后，追随者才可以进行新的请求。

#### Zab选举机制

1）半数机制：集群中半数以上机器存活，集群可用。所以Zookeeper适合安装奇数台服务器。

2）Zookeeper虽然在配置文件中并没有指定Master和Slave。但是，Zookeeper工作时，是有一个节点为Leader，其他则为Follower，Leader是通过内部的选举机制临时产生的。

假设有五台服务器组成的Zookeeper集群，它们的id从1-5，同时它们都是最新启动的，也就是没有历史数据，在存放数据量这一点上，都是一样的。假设这些服务器依序启动：

1. 服务器1启动，发起一次选举。服务器1投自己一票。此时服务器1票数一票，不够半数以上（3票），选举无法完成，服务器1状态保持为**LOOKING**；
2. 服务器2启动，再发起一次选举。服务器1和2分别投自己一票并交换选票信息：此时服务器1发现服务器2的ID比自己目前投票推举的（服务器1）大，更改选票为推举服务器2。此时服务器1票数0票，服务器2票数2票，没有半数以上结果，选举无法完成，服务器1，2状态保持**LOOKING**；
3. 服务器3启动，发起一次选举。此时服务器1和2都会更改选票为服务器3。此次投票结果：服务器1为0票，服务器2为0票，服务器3为3票。此时服务器3的票数已经超过半数，服务器3当选Leader。服务器1，2更改状态为**FOLLOWING**，服务器3更改状态为**LEADING**；
4. 服务器4启动，发起一次选举。此时服务器1，2，3已经不是LOOKING状态，不会更改选票信息。交换选票信息结果：服务器3为3票，服务器4为1票。此时服务器4服从多数，更改选票信息为服务器3，并更改状态为**FOLLOWING**；
5. 服务器5启动，同4一样当小弟。

#### 观察者

观察者与追随者有一些共同的特点，他们提交来自群首的提议，不同于追随者的是，**观察者不参与选举过程**，他们仅仅学习经由 INFORM 消息提交的提议。

引入观察者的一个主要原因是**提高读请求的可扩展性**。通过加入多个观察者，我们可以**在不牺牲写操作的吞吐率的前提下服务更多的读操作**。但是引入观察者也不是完全没有开销，每一个新加入的观察者将对应于每一个已提交事务点引入的一条额外消息。

采用观察者的另外一个原因是进行**跨多个数据中心部署**。由于数据中心之间的网络链接延时，将服务器分散于多个数据中心将明显地降低系统的速度。引入观察者后，更新请求能够先以高吞吐量和低延迟的方式在一个数据中心内执行，接下来再传播到异地的其他数据中心得到执行。

### 请求、事务、标识符

ZooKeeper 服务器会在本地处理只读请求（exists、getData、getChildren），例如一个服务器接收客户端的 getData 请求，服务器读取该状态信息，并把这些信息返回给客户端。

那些会改变 ZooKeeper 状态的客户端请求（create，delete 和 setData）将会**转发到群首**，群首执行对应的请求，并形成状态的更新，称为**事务（transaction）**，其中事务要以**原子**方式执行。同时，一个事务还要具有**幂等性**，事务的幂等性在我们进行恢复处理时更加简单，可以利用幂等性进行数据恢复或者灾备。

在群首产生了一个事务，就会为该事务分配一个标识符，称为**会话 id（zxid）**，通过 Zxid 对事务进行标识，就可以按照群首所指定的顺序在各个服务器中按序执行。服务器之间在进行新的群首选举时也会交换 zxid 信息，这样就可以知道哪个无故障服务器接收了更多的事务，并可以同步他们之间的状态信息。

Zxid 为一个 long 型（64 位）整数，分为两部分：**时间戳（epoch），逻辑时钟**部分和**计数器（counter），事务id**部分。每一部分为 32 位，在**zab 协议**中会有逻辑时钟（epoch）和计数器（counter）的具体作用，通过 zab 协议来广播各个服务器的状态变更信息。每一轮选举完成，epoch递增。

### Paxos算法

Paxos算法一种**基于消息传递**且具有高度容错特性的一致性算法。

分布式系统中的节点通信存在两种模型：**共享内存（Shared memory）**和**消息传递（Messages passing）**。基于消息传递通信模型的分布式系统，不可避免的会发生以下错误：进程可能会慢、被杀死或者重启，消息可能会延迟、丢失、重复，在基础 Paxos 场景中，先不考虑可能出现**消息篡改即拜占庭错误**的情况。Paxos 算法解决的问题是在一个可能发生上述异常的分布式系统中如何就某个值达成一致，保证不论发生以上任何异常，都不会破坏决议的一致性。

在一个Paxos系统中，首先将所有节点划分为Proposers、Acceptors、Learners（每个节点都可以身兼数职）；

一个完整的Paxos算法流程分为三个阶段：

- **Prepare阶段：**
  - Proposer向Acceptors发出Prepare请求Promise（承诺）；
  - Acceptors针对收到的Prepare请求进行Promise承诺；
- **Accept阶段：**
  - Proposer收到多数Acceptors承诺的Promise后，向Acceptors发出Propose请求；
  - Acceptors针对收到的Propose请求进行Accept处理；
- **Learn阶段：**
  - Proposer将形成的决议发送给所有Learners；

流程中的每条消息描述如下：

1. **Prepare:** Proposer生成全局唯一且递增的Proposal ID (可使用时间戳加Server ID)，向所有Acceptors发送Prepare请求，这里无需携带提案内容，只携带Proposal ID即可。
2. **Promise:** Acceptors收到Prepare请求后，做出“**两个承诺，一个应答**”。
   - （**承诺**）不再接受Proposal ID小于等于（注意：这里是<= ）当前请求的Prepare请求。
   - （**承诺**）不再接受Proposal ID小于（注意：这里是< ）当前请求的Propose请求。
   - （**应答**）不违背以前做出的承诺下，回复已经Accept过的提案中Proposal ID最大的那个提案的Value和Proposal ID，没有则返回空值。

3. **Propose:** Proposer 收到多数Acceptors的Promise应答后，从应答中选择Proposal ID最大的提案的Value，作为本次要发起的提案。如果所有应答的提案Value均为空值，则可以自己随意决定提案Value。然后携带当前Proposal ID，向所有Acceptors发送Propose请求。

4. **Accept:** Acceptor收到Propose请求后，在不违背自己之前做出的承诺下，接受并持久化当前Proposal ID和提案Value。

5. **Learn:** Proposer收到多数Acceptors的Accept后，决议形成，将形成的决议发送给所有Learners。

Paxos算法缺陷：在网络复杂的情况下，一个应用Paxos算法的分布式系统，可能很久无法收敛，甚至陷入活锁的情况。

![1577021201059](/img/Zookeeper/paxos算法.png)

![1577021099781](/img/Zookeeper/paxos算法_活锁.png)

系统中有一个以上的Proposer，多个Proposers相互争夺Acceptors，会造成迟迟无法达成一致的情况。针对这种情况，一种改进的Paxos算法被提出：从系统中选出一个节点作为Leader，只有Leader能够发起提案。这样，一次Paxos流程中只有一个Proposer，不会出现活锁的情况，此时只会出现例子中第一种情况。

### Raft算法

- [raft算法动画演示](http://thesecretlivesofdata.com/raft/)

Paxos算法不容易实现，Raft算法是对Paxos算法的简化和改进。

1. **Leader**总统节点，负责发出提案
2. **Follower**追随者节点，负责同意Leader发出的提案
3. **Candidate**候选人，负责争夺Leader

Raft算法将一致性问题分解为两个的子问题，**Leader选举**和**状态复制**

- Leader选举：
  - 每个Follower都持有一个**定时器**；
  - 当定时器时间到了而集群中仍然没有Leader，Follower将声明自己是Candidate并参与Leader选举，同时**将消息发给其他节点来争取他们的投票**，若其他节点长时间没有响应Candidate将重新发送选举信息；
  - 集群中其他节点将给Candidate投票；
  - 获得多数派支持的Candidate将成为**第M任Leader**（M任是最新的任期）；
  - 在任期内的Leader会**不断发送心跳**给其他节点证明自己还活着，其他节点受到心跳以后就清空自己的计时器并回复Leader的心跳。这个机制保证其他节点不会在Leader任期内参加Leader选举。
  - 当Leader节点出现故障而导致Leader失联，没有接收到心跳的Follower节点将准备成为Candidate进入下一轮Leader选举；
  - 若出现两个Candidate同时选举并获得了相同的票数，那么这两个Candidate将**随机推迟一段时间**后再向其他节点发出投票请求，这保证了再次发送投票请求以后不冲突；
- 状态复制：
  - Leader负责接收来自Client的提案请求；
  - 提案内容将包含在Leader发出的**下一个心跳中**；
  - Follower接收到心跳以后回复Leader的心跳；
  - Leader接收到多数派Follower的回复以后**确认提案**并写入自己的存储空间中并**回复Client**；
  - Leader**通知Follower节点确认提案**并写入自己的存储空间，随后所有的节点都拥有相同的数据；
  - 若集群中出现网络异常，导致集群被分割，将出现多个Leader；
  - 被分割出的非多数派集群将无法达到共识，即**脑裂**；
  - 当集群再次连通时，将**只听从最新任期Leader**的指挥，旧Leader将退化为Follower，此时集群重新达到一致性状态；

### ZAB协议

ZAB协议：**ZooKeeper Atomic Broadcast**，**ZooKeeper 原子广播协议**。

- 没有leader选leader（崩溃恢复）；
- 有leader就干活（增删读写）。

#### 状态更新的广播协议

在接收到一个写请求操作后，追随者会将请求转发给群首，群首将会探索性的执行该请求，并将执行结果以事务的方式对状态更新进行广播。如何确认一个事务是否已经提交，ZooKeeper 由此引入了 zab 协议。该协议提交一个事务非常简单，类型于一个**两阶段提交(2pc)**。

1. 群首向所有追随者发送一个 **PROPOSAL** 消息 p。
2. 当一个追随者接收到消息 p 后，会响应群首一个 **ACK** 消息，通知群首其已接受该提案（proposal）。
3. 当收到仲裁数量的服务器发送的确认消息后（该仲裁数包括群首自己），群首就会发送消息通知追随者进行提交（**COMMIT**）操作。

**Zab 保障了以下几个重要的属性**

- 如果群首按顺序广播了事务 T 和事务 T，那么每个服务器在提交 T 事务前保证事务 T 已经完成提交。
- 如果某个服务器按照事务 T 和事务 T 的顺序提交了事务，所有其他服务器也必然会在提交事务 T 前提交事务 T。
- 第一个属性保证事务在服务器之间传送顺序的一致，而第二个竖向保证服务器不会跳过任何事务。

#### Zab 与 Raft 区别

1. 对于Leader的任期，raft叫做term，而ZAB叫做epoch；
2. 在状态复制的过程中，raft的心跳从Leader向Follower发送，而ZAB则相反；
3. zab是广播式互相计数方式，发现别人比自己牛逼的时候要帮助别人扩散消息，根据本机计数决定谁是主。raft是个节点发起投票，大家根据规则选择投与不投，然后各自收到别人对自己的投票超过半数时宣布自己成为主。

#### Zab 与 Paxos 的区别

Paxos 的思想在很多分布式组件中都可以看到，Zab 协议可以认为是基于 Paxos 算法实现的，先来看下两者之间的联系：

- 都存在一个 Leader 进程的角色，负责协调多个 Follower 进程的运行；
- 都应用 Quroum 机制，Leader 进程都会等待超过半数的 Follower 做出正确的反馈后，才会将一个提案进行提交；
- 在 Zab 协议中，Zxid 中通过 epoch 来代表当前 Leader 周期，在 Paxos 算法中，同样存在这样一个标识，叫做 Ballot Number；

两者之间的区别是，**Paxos是理论，Zab是实践**，Paxos是论文性质的，目的是设计一种通用的分布式一致性算法，而 Zab 协议应用在 ZooKeeper 中，是一个特别设计的崩溃可恢复的原子消息广播算法。

Zab 协议增加了崩溃恢复的功能，当 Leader 服务器不可用，或者已经半数以上节点失去联系时，ZooKeeper 会进入恢复模式选举新的 Leader 服务器，使集群达到一个一致的状态。

#### 写数据流程

![1577022020162](/img/Zookeeper/zookeeper_写数据流程.png)

### 总结

模型无论怎么变化，始终只有一个目的，那就是在一个 fault torlerance 的分布式架构下，如何尽量保证其整个系统的可用性和一致性。最理想的模型当然是 Paxos，然而理论到落地还是有差距的，所以诞生了 Raft 和 ZAB，虽然只有一个 leader，但我们允许 leader 挂掉可以重新选举 leader，这样，中心式和分布式达到了一个妥协。

## References

- [腾讯技术工程-浅谈 CAP 和 Paxos 共识算法](https://mp.weixin.qq.com/s?__biz=MjM5ODYwMjI2MA==&mid=2649745393&idx=1&sn=90f64ea82007b201cf58d1b7e24d28d3)
- [腾讯技术工程-ZooKeeper 源码和实践揭秘](https://mp.weixin.qq.com/s?__biz=MjM5ODYwMjI2MA==&mid=2649745966&idx=1&sn=50a6b9892783b9509c02ac0db0f4167e)
- [Raft算法动画演示](http://thesecretlivesofdata.com/raft/)
- [分布式一致性算法-Paxos、Raft、ZAB、Gossip](https://zhuanlan.zhihu.com/p/130332285)
- [深入浅出 ZooKeeper-vivo互联网技术](https://mp.weixin.qq.com/s?__biz=MzI4NjY4MTU5Nw==&mid=2247488750&idx=3&sn=046f8423d9622ed08737799d4b416d7b)
- [详解分布式协调服务 ZooKeeper](https://draveness.me/zookeeper-chubby/)
- [再谈基于 Kafka 和 ZooKeeper 的分布式消息队列原理](https://gitbook.cn/books/5bc446269a9adf54c7ccb8bc/index.html)
- [聊聊Zookeeper之会话机制Session](https://juejin.cn/post/6844904196580311053)
- [https://github.com/qiurunze123/zookeeperDesign](https://github.com/qiurunze123/zookeeperDesign)

