---
layout:     post
title:      "Spring Cloud 基础学习"
date:       2019-07-16 15:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - Spring Cloud
---


## Spring Cloud

### 0. 前言

#### 单体应用

##### 单体式应用的不足

1、一旦应用变成一个又大又复杂的怪物，敏捷开发和部署举步维艰，任何单个开发者都不可能搞懂它。

2、单体式应用也会降低开发速度。应用越大，启动时间会越长。

3、复杂而巨大的单体式应用也不利于持续性开发。

4、单体式应用在不同模块发生资源冲突时，扩展将会非常困难。

5、单体式应用另外一个问题是可靠性。任何一个模块中的一个bug将会影响到整个应用的可靠性。

6、单体式应用使得采用新架构和语言非常困难。

##### 单体应用六边形架构：

![img](/img/SpringCloud/六角形应用.png)

#### 微服务

微服务架构风格这种开发方法，是以开发一组小型服务的方式来开发一个独立的应用系统的。其中每个小型服务都运行在自己的进程中，并经常采用HTTP资源API这样轻量的机制来相互通信。这些服务围绕业务功能进行构建，并能通过全自动的部署机制来进行独立部署。这些微服务可以使用不同的语言来编写，并且可以使用不同的数据存储技术。对这些微服务我们仅做最低限度的集中管理。

一个微服务一般完成某个特定的功能，比如下单管理、客户管理等等。每一个微服务都是微型六角形应用，都有自己的业务逻辑和适配器。一些微服务还会发布API给其它微服务和应用客户端使用。其它微服务完成一个Web UI，运行时，每一个实例可能是一个云VM或者是Docker容器。



总的来说，微服务的主旨是将一个原本独立的系统拆分成多个小型服务，这些小型服务都在各自独立的进程中运行，服务之间通过基于HTTP的RESTful API进行通信协作，并且每个服务都维护着自身的数据存储、业务开发、自动化测试以及独立部署机制。

##### 微服务的特征

1、每个微服务可独立运行在自己的进程里；

2、一系列独立运行的微服务共同构建起了整个系统；

3、每个服务为独立的业务开发，一个微服务一般完成某个特定的功能

4、微服务之间通过一些轻量的通信机制进行通信，例如通过REST API或者RPC的方式进行调用。

##### 微服务架构的不足

1、微服务应用是分布式系统，由此会带来固有的复杂性。

2、分区的数据库架构。需要更新不同服务所使用的不同的数据库。

3、测试一个基于微服务架构的应用也是很复杂的任务。同样的服务测试需要启动和它有关的所有服务。

4、服务架构模式应用的改变将会波及多个服务。比如需要修改服务A、B、C，而A依赖B，B依赖C。

5、部署一个微服务应用也很复杂。许多需要配置、部署、扩展和监控的部分。



###  1. spring cloud：

官网介绍：

**COORDINATE ANYTHING: DISTRIBUTED SYSTEMS SIMPLIFIED**

​	Building distributed systems doesn't need to be complex and error-prone. Spring Cloud offers a simple and accessible programming model to the most common distributed system patterns, helping developers build resilient, reliable, and coordinated applications. Spring Cloud is built on top of Spring Boot, making it easy for developers to get started and become productive quickly.

![img](/img/SpringCloud/diagram-distributed-systems.svg)



springcloud是微服务架构的集大成者，将一系列优秀的组件进行了整合，基于springboot构建。

通过一些简单的注解，我们就可以快速的在应用中配置一下常用模块并构建庞大的分布式系统。

**spring cloud组件：**

![img](/img/SpringCloud/springcloud_组件.png)



独自启动不需要依赖其它组件：

- Eureka，服务注册中心，特性有失效剔除、服务保护。

- Dashboard，Hystrix仪表盘，监控集群模式和单点模式，其中集群模式需要收集器Turbine配合。

- Zuul，API服务网关，功能有路由分发和过滤。

- Spring Cloud Config，分布式配置中心，支持本地仓库、SVN、Git、Jar包内配置等模式，



融合在每个微服务中、依赖其它组件并为其提供服务:

- Ribbon，客户端负载均衡，特性有区域亲和、重试机制。

- Hystrix，客户端容错保护，特性有服务降级、服务熔断、请求缓存、请求合并、依赖隔离。

- Feign，声明式服务调用，本质上就是Ribbon+Hystrix

- Stream，消息驱动，有Sink、Source、Processor三种通道，特性有订阅发布、消费组、消息分区。

- Bus，消息总线，配合Config仓库修改的一种Stream实现，

- Sleuth，分布式服务追踪，需要搞清楚TraceID和SpanID以及抽样，如何与ELK整合。



**每个组件都不是平白无故的产生的，是为了解决某一特定的问题而存在。**

1、Eureka和Ribbon，是最基础的组件，一个注册服务，一个消费服务。

2、Hystrix为了优化Ribbon、防止整个微服务架构因为某个服务节点的问题导致崩溃，是个保险丝的作用。

3、Dashboard给Hystrix统计和展示用的，而且监控服务节点的整体压力和健康情况。

4、Turbine是集群收集器，服务于Dashboard的。

5、Feign是方便我们程序员写更优美的代码的。

6、Zuul是加在整个微服务最前沿的防火墙和代理器，隐藏微服务结点IP端口信息，加强安全保护的。

7、Config是为了解决所有微服务各自维护各自的配置，设置一个统一的配置中心，方便修改配置的。

8、Bus是因为config修改完配置后各个结点都要refresh才能生效实在太麻烦，所以交给bus来通知服务节点刷新配置的。

9、Stream是为了简化研发人员对MQ使用的复杂度，弱化MQ的差异性，达到程序和MQ松耦合。

10、Sleuth是因为单次请求在微服务节点中跳转无法追溯，解决任务链日志追踪问题的。

#### 1.1 Eureka
作用：实现服务治理（服务注册与发现）。

![img](/img/SpringCloud/Eureka.png)

由两个组件组成：Eureka服务端和Eureka客户端。
Eureka服务端用作服务注册中心。支持集群部署。
Eureka客户端是一个java客户端，用来处理服务注册与发现。

在应用启动时，Eureka客户端向服务端注册自己的服务信息，同时将服务端的服务信息缓存到本地。客户端会和服务端周期性的进行心跳交互，以更新服务租约和服务信息。

 `Eureka Server` 做成高可用的集群模式时，`Eureka Server` 采用的是 `Peer to Peer` 对等通信，这是一种去中心化的架构，没有 `Master/Slave` 的概念，节点之间通过互相注册来提高注册中心集群的可用性。

Eureka 的自我保护模式：当注册中心每分钟收到心跳续约数量低于一个阀值，instance的数量(60/每个instance的心跳间隔秒数) < 自我保护系数，就会触发自我保护。不再注销任何服务实例，默认的自我保护系数为0.85。

#### 1.2 Ribbon

作用：主要提供客户侧的软件负载均衡算法。

![img](/img/SpringCloud/Ribbon.png)

Ribbon是一个基于HTTP和TCP的客户端负载均衡工具，它基于Netflix Ribbon实现。通过Spring Cloud的封装，可以让我们轻松地将面向服务的REST模版请求自动转换成客户端负载均衡的服务调用。

如上图，关键点就是将外界的rest调用，根据负载均衡策略转换为微服务调用。

#### 1.3 Hystrix

作用：断路器，保护系统，控制故障范围。

![img](/img/SpringCloud/Hystrix.png)

在微服务架构中，服务与服务之间可以相互调用（RPC），在Spring Cloud可以用RestTemplate+Ribbon和Feign来调用。为了保证其高可用，单个服务通常会集群部署。由于网络原因或者自身的原因，如果单个服务出现问题，调用这个服务就会出现线程阻塞，此时若有大量的请求涌入，Servlet容器的线程资源会被消耗完毕，导致服务瘫痪。服务与服务之间的依赖性，故障会传播，会对整个微服务系统造成灾难性的严重后果，这就是服务故障的“雪崩”效应。

#### 1.4 Feign

作用：服务消费者

Feign是一个声明式的Web Service客户端，它的目的就是让Web Service调用更加简单。Feign提供了HTTP请求的模板，通过编写简单的接口和插入注解，就可以定义好HTTP请求的参数、格式、地址等信息。

Feign 默认集成了 Ribbon，并与Eureka结合，默认实现了负载均衡的效果，是基于接口的注解。同样注册到 Eureka Server 中去。在启动类加上 `@EnableFeignClients`注解 ，在接口处指定 `@FeignClient(value = "调用的服务名称")` ，加上Ribbon 完成 对别调用服务的负载均衡效果。如果再加上 ,`fallback = SchedualServiceHiHystric.class `就可以完成断路器 Hystrix 的功能。

#### 1.5 Zuul

作用：api网关，路由，负载均衡等多种作用

![img](/img/SpringCloud/Zuul.png)

类似nginx，反向代理的功能，不过netflix自己增加了一些配合其他组件的特性。

在微服务架构中，后端服务往往不直接开放给调用端，而是通过一个API网关根据请求的url，路由到相应的服务。当添加API网关后，在第三方调用端和服务提供方之间就创建了一面墙，这面墙直接与调用方通信进行权限控制，后将请求均衡分发给后台服务端。

例：

```
zuul:
  routes:
    api-a:
      path: /api-a/**
      serviceId: service-ribbon
    api-b:
      path: /api-b/**
      serviceId: service-feign
```
`api-a`的路由会分发给 `service-ribbon` ，而 `api-b`的路由会分发给 `service-feign` 。

实现 `ZuulFilter`接口，可以定义参数验证，做安全验证功能。

#### 1.6 Config

作用：配置管理

![img](/img/SpringCloud/Config.png)

SpringCloud Config提供服务器端和客户端。服务器存储后端的默认实现使用git，因此它轻松支持标签版本的配置环境，以及可以访问用于管理内容的各种工具。

这个还是静态的，得配合Spring Cloud Bus实现动态的配置更新。



### References：

0. [https://spring.io/](https://spring.io/)

1. [Introduction to Microservices](https://www.nginx.com/blog/introduction-to-microservices/)

   [（译1）微服务架构的优势与不足](http://dockone.io/article/394)

2. [Building Microservices: Using an API Gateway](https://www.nginx.com/blog/building-microservices-using-an-api-gateway/)

   [（译2）使用API Gateway](http://dockone.io/article/482)

3. [Building Microservices: Inter-Process Communication in a Microservices Architecture](https://www.nginx.com/blog/building-microservices-inter-process-communication/)

   [（译3）深入微服务架构的进程间通信](http://dockone.io/article/549)

4. [Service Discovery in a Microservices Architecture (this article)](https://www.nginx.com/blog/service-discovery-in-a-microservices-architecture/)

   [（译4）服务发现的可行方案以及实践案例](http://dockone.io/article/771)

5. [Event-Driven Data Management for Microservices](https://www.nginx.com/blog/event-driven-data-management-microservices/)

   [（译5）微服务的事件驱动数据管理](http://dockone.io/article/936)

6. [Choosing a Microservices Deployment Strategy](https://www.nginx.com/blog/deploying-microservices/)

   [（译6）选择微服务部署策略](http://dockone.io/article/1066)

7. [Refactoring a Monolith into Microservices](https://www.nginx.com/blog/refactoring-a-monolith-into-microservices/)

   [（译7）从单体式架构迁移到微服务架构](http://dockone.io/article/1266)

2. [Spring Cloud各组件总结归纳](https://blog.csdn.net/yejingtao703/article/details/78331442/)

3. [SpringCloud简介与5大常用组件](https://baijiahao.baidu.com/s?id=1621651597363566701&wfr=spider&for=pc)

