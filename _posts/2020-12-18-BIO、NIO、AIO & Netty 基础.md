---
layout:     post
title:      "BIO、NIO、AIO & Netty 基础"
date:       2020-12-18 19:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - Netty
    - Java
    - BBigData
---


## 一、I/O模型

I/O 模型简单的理解：就是用什么样的通道进行数据的发送和接收，很大程度上决定了程序通信的性能。

Java共支持3种网络编程模型/IO模式：BIO、NIO、AIO。

- **Java BIO** ： 同步并阻塞(传统阻塞型)，服务器实现模式为**一个连接一个线程**，即客户端有连接请求时服务器端就需要启动一个线程进行处理，如果这个连接不做任何事情会造成不必要的线程开销
- **Java NIO** ： 同步非阻塞，服务器实现模式为**一个线程处理多个请求(连接)**，即客户端发送的连接请求都会注册到多路复用器上，多路复用器轮询到连接有I/O请求就进行处理
- **Java AIO(NIO.2)** ： 异步非阻塞，AIO 引入**异步通道**的概念，采用了 Proactor 模式，简化了程序编写，有效的请求才启动线程，它的特点是先由操作系统完成后才通知服务端程序启动线程去处理，一般适用于连接数较多且连接时间较长的应用

BIO、NIO、AIO适用场景分析：

- BIO方式适用于连接数目比较小且固定的架构，这种方式对服务器资源要求比较高，并发局限于应用中，JDK1.4以前的唯一选择，但程序简单易理解。
- NIO方式适用于连接数目多且连接比较短（轻操作）的架构，比如聊天服务器，弹幕系统，服务器间通讯等。编程比较复杂，JDK1.4开始支持。
- AIO方式使用于连接数目多且连接比较长（重操作）的架构，比如相册服务器，充分调用OS参与并发操作，编程比较复杂，JDK7开始支持。

## 二、unixIO

### 背景知识

#### 同步、异步、阻塞和非阻塞

首先需要有以下的清晰认知：

- **阻塞操作不等于同步**（blocking operation does NOT equal to synchronous）
- **非阻塞操作不等于异步**（non-blocking operation does NOT equal to asynchronous）
- 事实上，**同步异步与阻塞和非阻塞没有什么直接的关联关系**

**同步和异步：**

- 同步是指在发出一个function调用时，在没有得到结果之前，该调用就不返回。但是一旦调用返回，就得到调用结果了。这个结果可能是一个正确的期望结果，也可能是因为异常原因（比如超时）导致的失败结果。换句话说，就是由调用者主动等待这个调用的结果。
- 异步是调用在发出之后，本次调用过程就直接返回了，并没有同时返回结果。换句话说，当一个异步过程调用发出后，调用者不会立刻得到结果。而是在调用发出后，被调用者通过状态变化、事件通知等机制来通知调用者，或通过回调函数处理这个调用。

**阻塞和非阻塞：**

- 阻塞调用是指调用结果返回之前，当前线程会被挂起。调用线程只有在得到结果之后才会返回。
- 非阻塞是指在不能立刻得到结果之前，该调用不会阻塞当前线程。

#### 文件描述符fd

文件描述符（File descriptor）是计算机科学中的一个术语，是一个用于表述**指向文件的引用**的抽象化概念。

文件描述符在形式上是一个**非负整数**。实际上，它是一个**索引值**，指向内核为每一个进程所维护的该进程打开文件的记录表。当程序打开一个现有文件或者创建一个新文件时，内核向进程返回一个文件描述符。在程序设计中，一些涉及底层的程序编写往往会围绕着文件描述符展开。但是文件描述符这一概念往往只适用于UNIX、Linux这样的操作系统。

#### 用户空间与内核空间

User space（用户空间）和 Kernel space（内核空间）。

<img src="/img/Netty/unixIO-用户空间与核空间.png" alt="img" style="zoom: 67%;" />

简单说，Kernel space 是 Linux 内核的运行空间，User space 是用户程序的运行空间。为了安全，它们是隔离的，即使用户的程序崩溃了，内核也不受影响。

Kernel space 可以执行任意命令，调用系统的一切资源；User space 只能执行简单的运算，不能直接调用系统资源，必须通过系统接口（又称 system call），才能向内核发出指令。

```java
str = "my string" // 用户空间
x = x + 2 // 用户空间
file.write(str) // 切换到内核空间
y = x + 4 // 切换回用户空间
```

上面代码中，第一行和第二行都是简单的赋值运算，在 User space 执行。第三行需要写入文件，就要切换到 Kernel space，因为用户不能直接写文件，必须通过内核安排。第四行又是赋值运算，就切换回 User space。

#### 进程的阻塞

正在执行的进程，由于期待的某些事件未发生，如请求系统资源失败、等待某种操作的完成、新数据尚未到达或无新工作做等，则由系统自动执行阻塞原语(Block)，使自己由运行状态变为阻塞状态。可见，进程的阻塞是进程自身的一种主动行为，也因此只有处于运行态的进程（获得CPU），才可能将其转为阻塞状态。当进程进入阻塞状态，是不占用CPU资源的。

#### 进程切换

为了控制进程的执行，内核必须有能力挂起正在CPU上运行的进程，并恢复以前挂起的某个进程的执行。这种行为被称为进程切换。因此可以说，任何进程都是在操作系统内核的支持下运行的，是与内核紧密相关的。进程之间的切换其实是需要耗费cpu时间的。

#### 缓存I/O

缓存I/O又被称作标准I/O，大多数文件系统的默认I/O操作都是缓存I/O。在Linux的缓存I/O机制中，**数据先从磁盘复制到内核空间的缓冲区，然后从内核空间缓冲区复制到应用程序的地址空间**。

- **读操作**：操作系统检查内核的缓冲区有没有需要的数据，如果已经缓存了，那么就直接从缓存中返回；否则从磁盘中读取，然后缓存在操作系统的缓存中。
- **写操作**：将数据从用户空间复制到内核空间的缓存中。这时对用户程序来说写操作就已经完成，至于什么时候再写到磁盘中由操作系统决定，除非显示地调用了sync同步命令

**缓存I/O的优点：**

- 在一定程度上分离了内核空间和用户空间，保护系统本身的运行安全；
- 可以减少物理读盘的次数，从而提高性能。

**缓存I/O的缺点：**

- 在缓存I/O机制中，DMA方式可以将数据直接从磁盘读到页缓存中，或者将数据从页缓存直接写回到磁盘上，而不能直接在应用程序地址空间和磁盘之间进行数据传输，这样，数据在传输过程中需要在应用程序地址空间（用户空间）和缓存（内核空间）之间进行多次数据拷贝操作，这些数据拷贝操作所带来的CPU以及内存开销是非常大的。
- 因为这个原因的存在，所以又涉及到**零拷贝（zero copy）技术**。

### unixIO模型

在linux中，对于一次IO访问（以read举例），数据会先被拷贝到操作系统内核的缓冲区中，然后才会从操作系统内核的缓冲区拷贝到应用程序的地址空间。所以说，当一个read操作发生时，它会经历两个阶段：

- 等待数据准备就绪 (Waiting for the data to be ready)
- 将数据从内核拷贝到进程中 (Copying the data from the kernel to the process)

正式因为这两个阶段，linux系统产生了下面五种网络模式的方案：

- 阻塞式IO模型(blocking IO model)
- 非阻塞式IO模型(noblocking IO model)
- IO复用式模型(IO multiplexing model)
- 异步IO式模型(asynchronous IO model)
- 信号驱动式IO模型(signal-driven IO model)

#### 阻塞式IO模型

在linux中，默认情况下所有的IO操作都是blocking，一个典型的读操作流程大概是这样：

<img src="/img/Netty/unixIO-Blocking IO Model.png" alt="img" style="zoom: 33%;" />

当用户进程调用了recvfrom这个系统调用，kernel就开始了IO的第一个阶段：准备数据（对于网络IO来说，很多时候数据在一开始还没有到达。比如，还没有收到一个完整的UDP包。这个时候kernel就要**等待足够的数据到来**），而数据被拷贝到操作系统内核的缓冲区中是需要一个过程的，这个过程需要等待。而在用户进程这边，整个进程会被阻塞（当然，是进程自己选择的阻塞）。当kernel一直等到数据准备好了，它就会将数据从kernel中拷贝到用户空间的缓冲区以后，然后kernel返回结果，用户进程才解除block的状态，重新运行起来。

所以：blocking IO的特点就是在IO执行的下两个阶段的时候都被block了。

- 等待数据准备就绪 (Waiting for the data to be ready) 「阻塞」
- 将数据从内核拷贝到进程中 (Copying the data from the kernel to the process) 「阻塞」

####非阻塞 I/O

linux下，可以通过设置socket使其变为non-blocking。通过java可以这么操作：

```java
InetAddress host = InetAddress.getByName("localhost");
Selector selector = Selector.open();
ServerSocketChannel serverSocketChannel = ServerSocketChannel.open();
// non-blocking
serverSocketChannel.configureBlocking(false);
serverSocketChannel.bind(new InetSocketAddress(hos1234));
serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);
```

socket设置为 NONBLOCK（非阻塞）就是告诉内核，当所请求的I/O操作无法完成时，不要将进程睡眠，而是返回一个错误码(EWOULDBLOCK) ，这样请求就不会阻塞。

<img src="/img/Netty/unixIO-Nonblocking IO Model.png" alt="img" style="zoom: 50%;" />

当用户进程调用了recvfrom这个系统调用，如果kernel中的数据还没有准备好，那么它并不会block用户进程，而是**立刻返回一个EWOULDBLOCK error**。从用户进程角度讲 ，它发起一个read操作后，并不需要等待，而是马上就得到了一个结果。用户进程判断结果是一个EWOULDBLOCK error时，它就知道数据还没有准备好，于是它可以再次发送read操作。一旦kernel中的数据准备好了，并且又再次收到了用户进程的system call，那么它马上就将数据拷贝到了用户空间缓冲区，然后返回。

可以看到，I/O 操作函数将不断的测试数据是否已经准备好，如果没有准备好，继续轮询，直到数据准备好为止。整个 I/O 请求的过程中，**虽然用户线程每次发起 I/O 请求后可以立即返回，但是为了等到数据，仍需要不断地轮询、重复请求，消耗了大量的 CPU 的资源**。

所以，non blocking IO的特点是用户进程需要不断的主动询问kernel数据好了没有：

- 等待数据准备就绪 (Waiting for the data to be ready) 「非阻塞」
- 将数据从内核拷贝到进程中 (Copying the data from the kernel to the process) 「阻塞」

**一般很少直接使用这种模型，而是在其他 I/O 模型中使用非阻塞 I/O 这一特性。这种方式对单个 I/O 请求意义不大，但给 I/O 多路复用铺平了道路。**

#### I/O多路复用

IO multiplexing就是我们常说的**select**，**poll**，**epoll**，有些地方也称这种IO方式为event driven IO。select/epoll的好处就在于单个process就可以同时处理多个网络连接的IO。它的基本原理就是select，poll，epoll这些个function会不断的轮询所负责的所有socket，当某个socket有数据到达了，就通知用户进程。

<img src="/img/Netty/unixIO-IO Multiplexing Model.png" alt="img" style="zoom: 33%;" />

当用户进程调用了select，那么整个进程会被block，而同时，kernel会“监视”所有select负责的socket，当任何一个socket中的数据准备好了，select就会返回。这个时候用户进程再调用read操作，将数据从kernel拷贝到用户进程。

所以，I/O 多路复用的特点是通过一种机制一个进程能同时等待多个文件描述符，而这些文件描述符（套接字描述符）其中的任意一个进入读就绪状态，select()函数就可以返回。

这个图和blocking IO的图其实并没有太大的不同，**事实上因为IO多路复用多了添加监视 socket，以及调用 select 函数的额外操作，效率还更差一些。**因为这里需要使用两个system call (select 和 recvfrom)，而blocking IO只调用了一个system call (recvfrom)。但是，使用 select 以后**最大的优势是用户可以在一个线程内同时处理多个 socket 的 I/O 请求。**用户可以注册多个 socket，然后不断地调用 select 读取被激活的 socket，即可达到在同一个线程内同时处理多个 I/O 请求的目的。而在同步阻塞模型中，必须通过多线程的方式才能达到这个目的。

所以，如果处理的连接数不是很高的话，使用select/epoll的web server不一定比使用multi-threading + blocking IO的web server性能更好，可能延迟还更大。select/epoll的**优势并不是对于单个连接能处理得更快，而是在于能处理更多的连接。**

在IO multiplexing Model中，实际中，对于每一个socket，一般都设置成为non-blocking，但是，如上图所示，整个用户的process其实是一直被block的。只不过process是被select这个函数block，而不是被socket IO给block。

因此对于IO多路复用模型来说： 

- 等待数据准备就绪 (Waiting for the data to be ready) 「阻塞」
- 将数据从内核拷贝到进程中 (Copying the data from the kernel to the process) 「阻塞」

##### select、poll、epoll

> I/O多路复用（multiplexing）的本质是通过一种机制（系统内核缓冲I/O数据），让单个进程可以监视多个文件描述符，一旦某个描述符就绪（一般是读就绪或写就绪），能够通知程序进行相应的读写操作

select、poll 和 epoll 都是 Linux API 提供的 IO 复用方式。

###### Select

select函数：

```c
int select(int maxfdp1,fd_set *readset,fd_set *writeset,fd_set *exceptset,const struct timeval *timeout);
```

**【参数说明】**
 **int maxfdp1**：指定待测试的文件描述字个数，它的值是待测试的最大描述字加1。
 **fd_set \*readset , fd_set \*writeset , fd_set \*exceptset**： `fd_set`可以理解为一个集合，这个集合中存放的是文件描述符(file descriptor)，即文件句柄。中间的三个参数指定要让内核测试读、写和异常条件的文件描述符集合。如果对某一个的条件不感兴趣，就可以把它设为空指针。
 **const struct timeval \*timeout** ：`timeout`告知内核等待所指定文件描述符集合中的任何一个就绪可花多少时间。其timeval结构用于指定这段时间的秒数和微秒数。

**【返回值】**
 **int** ：若有就绪描述符返回其数目，若超时则为0，若出错则为-1

**select运行机制**

select()的机制中提供一种`fd_set`的数据结构，实际上是一个long类型的数组，每一个数组元素都能与一打开的文件句柄（不管是Socket句柄,还是其他文件或命名管道或设备句柄）建立联系，建立联系的工作由程序员完成，当调用select()时，由内核根据IO状态修改fd_set的内容，由此来通知执行了select()的进程哪一Socket或文件可读。

从流程上来看，使用select函数进行IO请求和同步阻塞模型没有太大的区别，甚至还多了添加监视socket，以及调用select函数的额外操作，效率更差。但是，使用select以后最大的优势是用户可以在一个线程内同时处理多个socket的IO请求。用户可以注册多个socket，然后不断地调用select读取被激活的socket，即可达到在同一个线程内同时处理多个IO请求的目的。而在同步阻塞模型中，必须通过多线程的方式才能达到这个目的。

**select机制的问题**

1. 每次调用select，都需要把`fd_set`集合从用户态拷贝到内核态，如果`fd_set`集合很大时，那这个开销也很大；
2. 同时每次调用select都需要在内核遍历传递进来的所有`fd_set`，如果`fd_set`集合很大时，那这个开销也很大；
3. 为了减少数据拷贝带来的性能损坏，内核对被监控的`fd_set`集合大小做了限制，并且这个是通过宏控制的，大小不可改变(限制为1024)。

###### Poll

poll的机制与select类似，与select在本质上没有多大差别，管理多个描述符也是进行轮询，根据描述符的状态进行处理，但是poll没有最大文件描述符数量的限制（链表）。也就是说，poll只解决了上面的问题3，并没有解决问题1，2的性能开销问题。

下面是pll的函数原型：

```cpp
int poll(struct pollfd *fds, nfds_t nfds, int timeout);

typedef struct pollfd {
        int fd;                         // 需要被检测或选择的文件描述符
        short events;                   // 对文件描述符fd上感兴趣的事件
        short revents;                  // 文件描述符fd上当前实际发生的事件
} pollfd_t;
```

poll改变了文件描述符集合的描述方式，使用了`pollfd`结构而不是select的`fd_set`结构，使得poll支持的文件描述符集合限制远大于select的1024。

**【参数说明】**

**struct pollfd \*fds** ：`fds`是一个`struct pollfd`类型的数组，用于存放需要检测其状态的socket描述符，并且调用poll函数之后`fds`数组不会被清空；一个`pollfd`结构体表示一个被监视的文件描述符，通过传递`fds`指示 poll() 监视多个文件描述符。其中，结构体的`events`域是监视该文件描述符的事件掩码，由用户来设置这个域，结构体的`revents`域是文件描述符的操作结果事件掩码，内核在调用返回时设置这个域。

**nfds_t nfds** ：记录数组`fds`中描述符的总数量。

**【返回值】**
 **int** ：函数返回fds集合中就绪的读、写，或出错的描述符数量，返回0表示超时，返回-1表示出错；

###### Epoll

epoll在Linux2.6内核正式提出，是基于事件驱动的I/O方式，相对于select来说，epoll没有描述符个数限制，使用一个文件描述符管理多个描述符，将用户关心的文件描述符的事件存放到内核的一个事件表中，这样在用户空间和内核空间的copy只需一次。

Linux中提供的epoll相关函数如下：

```csharp
int epoll_create(int size);
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);
int epoll_wait(int epfd, struct epoll_event * events, int maxevents, int timeout);
```

**1. epoll_create**：函数创建一个epoll句柄，参数`size`表明内核要监听的描述符数量。调用成功时返回一个epoll句柄描述符，失败时返回-1。

**2. epoll_ctl**：函数注册要监听的事件类型。四个参数解释如下：

- `epfd` 表示epoll句柄
- `op` 表示fd操作类型，有如下3种：
  - EPOLL_CTL_ADD   注册新的fd到epfd中
  - EPOLL_CTL_MOD 修改已注册的fd的监听事件
  - EPOLL_CTL_DEL 从epfd中删除一个fd
- `fd` 是要监听的描述符
- `event` 表示要监听的事件

epoll_event 结构体定义如下：

```cpp
struct epoll_event {
    __uint32_t events;  /* Epoll events */
    epoll_data_t data;  /* User data variable */
};

typedef union epoll_data {
    void *ptr;
    int fd;
    __uint32_t u32;
    __uint64_t u64;
} epoll_data_t;
```

**3. epoll_wait**：函数等待事件的就绪，成功时返回就绪的事件数目，调用失败时返回 -1，等待超时返回 0。

- `epfd` 是epoll句柄
- `events` 表示从内核得到的就绪事件集合
- `maxevents` 告诉内核events的大小
- `timeout` 表示等待的超时事件

epoll是Linux内核为处理大批量文件描述符而作了改进的poll，是Linux下多路复用IO接口select/poll的增强版本，它能显著提高程序在大量并发连接中只有少量活跃的情况下的系统CPU利用率。原因就是获取事件的时候，它无须遍历整个被侦听的描述符集，只要遍历那些被内核IO事件异步唤醒而加入Ready队列的描述符集合就行了。

epoll除了提供select/poll那种IO事件的水平触发（Level Triggered）外，还提供了边缘触发（Edge Triggered），这就使得用户空间程序有可能缓存IO状态，减少epoll_wait/epoll_pwait的调用，提高应用程序效率。

- **水平触发（LT）：**默认工作模式，即当epoll_wait检测到某描述符事件就绪并通知应用程序时，应用程序可以不立即处理该事件；下次调用epoll_wait时，会再次通知此事件
- **边缘触发（ET）：** 当epoll_wait检测到某描述符事件就绪并通知应用程序时，应用程序必须立即处理该事件。如果不处理，下次调用epoll_wait时，不会再次通知此事件。（直到你做了某些操作导致该描述符变成未就绪状态了，也就是说边缘触发只在状态由未就绪变为就绪时只通知一次）。

LT和ET原本应该是用于脉冲信号的，可能用它来解释更加形象。Level和Edge指的就是触发点，Level为只要处于水平，那么就一直触发，而Edge则为上升沿和下降沿的时候触发。比如：0->1 就是Edge，1->1 就是Level。

ET模式很大程度上减少了epoll事件的触发次数，因此效率比LT模式下高。

###### 对比

select,poll,epoll的区别：

|            |                       select                       |                       poll                       |                            epoll                             |
| :--------- | :------------------------------------------------: | :----------------------------------------------: | :----------------------------------------------------------: |
| 操作方式   |                        遍历                        |                       遍历                       |                             回调                             |
| 底层实现   |                        数组                        |                       链表                       |                            红黑树                            |
| IO效率     |    每次调用都进行线性遍历，时间复杂度为**O(n)**    |   每次调用都进行线性遍历，时间复杂度为**O(n)**   | 事件通知方式，每当fd就绪，系统注册的回调函数就会被调用，将就绪fd放到readyList里面，时间复杂度**O(1)** |
| 最大连接数 |              1024（x86）或2048（x64）              |                      无上限                      |                            无上限                            |
| fd拷贝     | 每次调用select，都需要把fd集合从用户态拷贝到内核态 | 每次调用poll，都需要把fd集合从用户态拷贝到内核态 |  调用epoll_ctl时拷贝进内核并保存，之后每次epoll_wait不拷贝   |

epoll是Linux目前大规模网络并发程序开发的首选模型。在绝大多数情况下性能远超select和poll。目前流行的高性能web服务器Nginx正式依赖于epoll提供的高效网络套接字轮询服务。但是，在并发连接不高的情况下，多线程+阻塞I/O方式可能性能更好。

#### 异步I/O

linux下的asynchronous IO的流程：

<img src="/img/Netty/unixIO-Asynchronous IO Model.png" alt="img" style="zoom: 50%;" />

**用户进程发起aio_read调用之后，立刻就可以开始去做其它的事。**而另一方面，从kernel的角度，当它发现一个asynchronous read之后，首先它会立刻返回，所以不会对用户进程产生任何block。然后，kernel会等待数据准备完成，然后将数据拷贝到用户内存，当这**一切都完成之后，kernel会给用户进程发送一个signal，告诉它read操作完成了。**

异步 I/O 模型使用了 Proactor 设计模式实现了这一机制。

因此对异步IO模型来说：

- 等待数据准备就绪 (Waiting for the data to be ready) 「非阻塞」
- 将数据从内核拷贝到进程中 (Copying the data from the kernel to the process) 「非阻塞」

#### 信号驱动式IO模型

首先我们允许 socket 进行信号驱动 I/O,并安装一个信号处理函数，进程继续运行并不阻塞。当数据准备好时，进程会收到一个SIGIO信号，可以在信号处理函数中调用 I/O 操作函数处理数据。

<img src="/img/Netty/unixIO-Signal Driven IO Model.png" alt="img" style="zoom:50%;" />

但是这种IO模确用的不多。

#### 对比

| I/O模型                                  | 等待数据准备就绪 | 将数据从内核拷贝到进程中 |
| ---------------------------------------- | ---------------- | ------------------------ |
| 阻塞式IO模型(blocking IO model)          | 阻塞             | 阻塞                     |
| 非阻塞式IO模型(noblocking IO model)      | 非阻塞           | 阻塞                     |
| IO复用式模型(IO multiplexing model)      | 阻塞             | 阻塞                     |
| 异步IO式模型(asynchronous IO model)      | 非阻塞           | 非阻塞                   |
| 信号驱动式IO模型(signal-driven IO model) | 非阻塞           | 非阻塞                   |

### Reactor和Proactor

- reactor：能收了你跟俺说一声。
- proactor: 你给我收十个字节，收好了跟俺说一声。

#### Reactor 模式

针对传统阻塞 I/O 服务模型的 2 个缺点，解决方案：

- 基于 I/O 复用模型：多个连接共用一个阻塞对象，应用程序只需要在一个阻塞对象等待，无需阻塞等待所有连接。当某个连接有新的数据可以处理时，操作系统通知应用程序，线程从阻塞状态返回，开始进行业务处理Reactor 对应的叫法: 
  - 1.反应器模式 
  - 2.分发者模式(Dispatcher) 
  - 3.通知者模式(notifier)
- 基于线程池复用线程资源：不必再为每个连接创建线程，将连接完成后的业务处理任务分配给线程进行处理，一个线程可以处理多个连接的业务。

Reactor 翻译过来的意思是「反应堆」，这里的反应指的是「**对事件反应**」，也就是**来了一个事件，Reactor 就有相对应的反应/响应**。

**I/O 复用结合线程池**，就是 Reactor 模式基本设计思想，如图：

<img src="/img/Netty/Reactor模式.png" alt="img" style="zoom:67%;" />

- Reactor 模式，通过一个或多个输入同时传递给服务处理器的模式(基于事件驱动)；
- 服务器端程序处理传入的多个请求,并将它们同步分派到相应的处理线程， 因此Reactor模式也叫 Dispatcher模式；
- Reactor 模式使用IO复用监听事件, 收到事件后，分发给某个线程(进程), 这点就是网络服务器高并发处理关键。

Reactor 三种模式：

- 单 Reactor 单线程：前台接待员和服务员是同一个人，全程为顾客服；
- 单 Reactor 多线程：1 个前台接待员，多个服务员，接待员只负责接待；
- 主从 Reactor 多线程：多个前台接待员，多个服务生；

#### Proactor模式

 Reactor 是非阻塞同步网络模式，而 **Proactor 是异步网络模式**。

- **Reactor 是非阻塞同步网络模式，感知的是就绪可读写事件**。在每次感知到有事件发生（比如可读就绪事件）后，就需要应用进程主动调用 read 方法来完成数据的读取，也就是要应用进程主动将 socket 接收缓存中的数据读到应用进程内存中，这个过程是同步的，读取完数据后应用进程才能处理数据。
- **Proactor 是异步网络模式， 感知的是已完成的读写事件**。在发起异步读写请求时，需要传入数据缓冲区的地址（用来存放结果数据）等信息，这样系统内核才可以自动帮我们把数据的读写工作完成，这里的读写工作全程由操作系统来做，并不需要像 Reactor 那样还需要应用进程主动发起 read/write 来读写数据，操作系统完成读写工作后，就会通知应用进程直接处理数据。

因此，**Reactor 可以理解为「来了事件操作系统通知应用进程，让应用进程来处理」**，而 **Proactor 可以理解为「来了事件操作系统来处理，处理完再通知应用进程」**。这里的「事件」就是有新连接、有数据可读、有数据可写的这些 I/O 事件这里的「处理」包含从驱动读取到内核以及从内核读取到用户空间。

无论是 Reactor，还是 Proactor，都是一种基于「事件分发」的网络编程模式，区别在于 **Reactor 模式是基于「待完成」的 I/O 事件，而 Proactor 模式则是基于「已完成」的 I/O 事件**。


## 三、BIO

Java BIO 就是传统的java io 编程，其相关的类和接口在 java.io BIO(blocking I/O) 。

同步阻塞，服务器实现模式为**一个连接一个线程**，即客户端有连接请求时服务器端就需要启动一个线程进行处理，如果这个连接不做任何事情会造成不必要的线程开销，可以通过线程池机制改善(实现多个客户连接服务器)。 

BIO编程简单**流程**：

就是任何一个网络通信框架都是一样的，就好比tomcat，基本流程：

- 服务器端启动一个ServerSocket；
- 客户端启动Socket对服务器进行通信，默认情况下服务器端需要对每个客户 建立一个线程与之通讯；
- 客户端发出请求后, 先咨询服务器是否有线程响应，如果没有则会等待，或者被拒绝；
- 如果有响应，客户端线程会等待请求结束后，再继续执行。

BIO这种IO模型有什么**问题**：

- 每个请求都需要创建独立的线程，与对应的客户端进行数据 Read，业务处理，数据 Write 。
- 当并发数较大时，需要创建大量线程来处理连接，系统资源占用较大。
- 连接建立后，如果当前线程暂时没有数据可读，则线程就阻塞在 Read 操作上，造成线程资源浪费。

## 四、NIO

### 简介

特点：

- NIO是 **面向缓冲区** ，或者面向 **块** 编程的。数据读取到一个它稍后处理的缓冲区，需要时可在缓冲区中前后移动，这就增加了处理过程中的灵活性，使用它可以提供非阻塞式的高伸缩性网络。
- Java NIO的**非阻塞模式**，使一个线程从某通道发送请求或者读取数据，但是它仅能得到目前可用的数据，如果目前没有数据可用时，就什么都不会获取，而不是保持线程阻塞，所以直至数据变的可以读取之前，该线程可以继续做其他的事情。 非阻塞写也是如此，一个线程请求写入一些数据到某通道，但不需要等待它完全写入，这个线程同时可以去做别的事情。

通俗理解：NIO是可以做到用一个线程来处理多个操作的。假设有10000个请求过来,根据实际情况，可以分配50或者100个线程来处理。不像之前的阻塞IO那样，非得分配10000个。

HTTP2.0使用了多路复用的技术，做到同一个连接并发处理多个请求，而且并发请求的数量比HTTP1.1大了好几个数量级。

**NIO 和 BIO 的比较**：

- BIO 以流的方式处理数据,而 NIO 以块的方式处理数据,块 I/O 的效率比流 I/O 高很多；
- BIO 是阻塞的，NIO 则是非阻塞的；
- BIO基于**字节流和字符流**进行操作，而 NIO 基于 **Channel(通道)和 Buffer(缓冲区)**进行操作，数据总是从通道读取到缓冲区中，或者从缓冲区写入到通道中。Selector(选择器)用于监听多个通道的事件（比如：连接请求，数据到达等），因此使用单个线程就可以监听多个客户端通道。

### 三大核心

NIO 有三大核心部分：

- **Buffer(缓冲区)**
- **Channel(通道)**
- **Selector(选择器)**

<img src="/img/Netty/NIO-三大核心组件.png" alt="NIO - Selector in Java network programming" style="zoom: 50%;" />

特点：

- 每个channel 都会对应一个Buffer；
- Selector 对应一个线程， 一个线程对应多个channel(连接)；
- 程序切换到哪个channel 是由事件决定的, Event 是一个重要的概念（EventLoop）；
- Selector 会根据不同的事件，在各个通道上切换；
- Buffer 就是一个内存块 ， 底层是有一个数组；
- 数据的读取写入是通过Buffer, 这个和BIO不同 , BIO 中要么是输入流，或者是输出流，不能双向，但是NIO的Buffer 是可以读也可以写, 需要 flip 方法切换；
- channel 是双向的, 可以返回底层操作系统的情况, 比如Linux， 底层的操作系统通道就是双向的。

实际上，NIO的组件和类远不止这三个，但这个三个组件是核心。至于其它组件，例如Pipe和FileLock只不过是围绕在三大核心组件旁的实用工具类。

#### Buffer

缓冲区本质上是一个可以读写数据的内存块，可以理解成是一个容器对象(含数组)，该对象提供了一组方法，可以更轻松地使用内存块，缓冲区对象内置了一些机制，能够跟踪和记录缓冲区的状态变化情况。Channel 提供从文件、网络读取数据的渠道，但是读取或写入的数据都必须经由 Buffer。**相比较直接对数组的操作。Buffer API 更加容易操作和管理。**

##### 缓存类型

Buffer 是一个顶层父类，它是一个抽象类，主要的缓存类型如下:

- ByteBuffer：字节缓存
- ShortBuffer：短整型缓存
- IntBuffer：整型缓存
- LongBuffer：长整型缓存
- FloatBuffer：单精度浮点类型缓存
- DoubleBuffer：双精度浮点类型缓存
- CharBuffer：字符缓存

这些缓存涵盖了你可以通过IO发送的基本数据类型：byte, short, int, long, float, double 和 char。此外，还有 MappedByteBuffer （映射字节缓存）可用来处理内存映射文件（memory mapped files）。

##### 数据读写四个步骤

使用Buffer进行数据写入与读取，需要进行如下**四个步骤**

- 将数据写入缓冲区。
- 调用buffer.flip()，转换为读取模式。
- 缓冲区读取数据。
- 调用buffer.clear() 或 buffer.compact() 消除缓冲区。

##### 四个属性

Buffer类定义了所有的缓冲区都具有的四个属性来提供关于其所包含的数据元素的信息：

| 属性     | 描述                                                         |
| -------- | ------------------------------------------------------------ |
| Capacity | 容量，即可以容纳的最大数据量；在缓冲区创建时被设定且不能改变。 |
| Limit    | 限制，写入模式，限制等于buffer的容量，读取模式下，limit等于写入的数据量。表示缓冲区的当前终点，不能对缓冲区超过Limit的位置进行读写操作，且Limit是可以修改的。 |
| Position | 位置，写入模式时代表写数据的位置。读取模式时代表读取数据的位置。每次读写缓冲区数时都会改变该值，为下次读写作准备。 |
| Mark     | 标记。                                                       |

![img](/img/Netty/NIO-Buffer.png)

##### 常用的api

```java
public abstract class Buffer {
    // 四个属性
    // Invariants: mark <= position <= limit <= capacity
    private int mark = -1;
    private int position = 0;
    private int limit;
    private int capacity;
    
    // 返回此缓冲区的容量
    public final int capacity() { return capacity;} 
    // 返回此缓冲区的位置
    public final int position() { return position;}
    // 设置此缓冲区的位置
    public final Buffer position(int newPosition) {
        if ((newPosition > limit) || (newPosition < 0))
            throw new IllegalArgumentException();
        position = newPosition;
        if (mark > position) mark = -1;
        return this;
    }
    // 返回此缓冲区的Limit
    public final int limit() { return limit;} 
    // 设置此缓冲区的Limit
    public final Buffer limit(int newLimit) {
        if ((newLimit > capacity) || (newLimit < 0))
            throw new IllegalArgumentException();
        limit = newLimit;
        if (position > limit) position = limit;
        if (mark > limit) mark = -1;
        return this;
    }
    // 在此缓冲区的位置设置标记
    public final Buffer mark() { mark = position; return this; }
    // 将此缓冲区的位置重置为之前标记的位置
    public final Buffer reset() {
        int m = mark;
        if (m < 0)
            throw new InvalidMarkException();
        position = m;
        return this;
    }
    
    // 清除此缓冲区，即将各个标记恢复到初始状态，但是数据并没有真正清除
    public final Buffer clear() {
        position = 0;
        limit = capacity;
        mark = -1;
        return this;
    }
    // 反转此缓冲区
    public final Buffer flip() {
        limit = position;
        position = 0;
        mark = -1;
        return this;
    }
    // 重绕此缓冲区，position=0，mark=-1
    public final Buffer rewind() {
        position = 0;
        mark = -1;
        return this;
    }
    // 返回当前位置与限制之间的元素数
    public final int remaining() { return limit - position;}
    // 告知当前位置和限制之间是否有元素
    public final boolean hasRemaining() { return position < limit; }
    // 告知此缓冲区是否只读缓冲区
    public abstract boolean isReadOnly();
    
    // JDK1.6引入API
    // 告知此缓冲区是否具有可以访问的底层实现数组
    public abstract boolean hasArray();
    // 告知此缓冲区的底层实现数组
    public abstract Object array();
    // 返回底层实现数组中第一个缓冲区元素的偏移量
    public abstract int arrayOffset();
    // 告知此缓冲区是否为直接缓冲区
    public abstract boolean isDirect();
}
```

#### Channel

NIO的通道类似于流（Stream），但有些区别如下：

- 通道**可以同时进行读写**，而流只能读或者只能写；
- 通道**可以实现异步读写数据**；
- 通道**可以从缓冲读数据，也可以写数据到缓冲**。
- BIO 中的 stream 是单向的，例如 FileInputStream 对象只能进行读取数据的操作，而 NIO 中的通道(Channel)是**双向**的，可以读操作，也可以写操作。

用IO的方式去读写数据的时候，读写是要分离的，即必须要明确是InputStream还是OutputStream，而在Channel这里，一条连接客户端和服务端的Channel是共用的，NIO开发中可以利用`channel.read()`方法去读取socket缓冲区的数据，也可以通过`channel.write()`去刷出数据到客户端。

流的read和write都是同步操作，在Stream中调用读写方法时，必须要等IO操作完成以后才能执行下一步，需要顺序执行而没有异步的方式可以用。而NIO中Channel的读写是可以设置为非阻塞的，非阻塞模式下，write()方法在尚未写出任何内容时可能就返回了，这种模式下须得在while循环中判断来调用write()。

##### 通道类型

主要的通道类型列表：

- FileChannel：文件通道
- DatagramChannel：数据报通道（UDP）
- SocketChannel：socket通道（TCP）
- ServerSocketChannel：serversocket通道（TCP）

##### FileChannel

FileChannel主要用来对本地文件进行 IO 操作，常见的方法有：

```java
// 从通道读取数据并放到缓冲区中
public int read(ByteBuffer dst) 
// 把缓冲区的数据写到通道中
public int write(ByteBuffer src) 
// 从目标通道中复制数据到当前通道
public long transferFrom(ReadableByteChannel src, long position, long count) 
// 把数据从当前通道复制给目标通道
public long transferTo(long position, long count, WritableByteChannel target) 
```

用FileChannel来读写文件：

```java
public class FileChannelTest {

    public static void main(String[] args) throws Exception {
        writeFile();
        readFile();
        readAndWriteFile();
    }

    /**
     * FileChannel写文件
     */
    private static void writeFile() throws Exception {
        String name = "hello,呱唧呱唧";

        //生成一个输出流
        FileOutputStream fileOutputStream = new FileOutputStream("d:\\tmp\\file01.txt");

        //通过 fileOutputStream 获取 对应的 FileChannel
        //这个 fileChannel 真实 类型是  FileChannelImpl
        FileChannel channel = fileOutputStream.getChannel();

        ByteBuffer byteBuffer = ByteBuffer.allocate(1024);
        //把数据放到缓存区中
        byteBuffer.put(name.getBytes());

        //转换并写出
        byteBuffer.flip();
        channel.write(byteBuffer);
        fileOutputStream.close();
    }

    /**
     * FileChannel 读文件
     */
    private static void readFile() throws Exception {
        //创建文件流
        File file = new File("d:\\tmp\\file01.txt");
        FileInputStream fileInputStream = new FileInputStream(file);

        // Buffer
        ByteBuffer allocate = ByteBuffer.allocate((int) file.length());

        // Channel
        FileChannel channel = fileInputStream.getChannel();
        channel.read(allocate);

        System.out.println(new String(allocate.array()));
        fileInputStream.close();
    }

    private static void readAndWriteFile() throws Exception {
        // 输入流
        FileInputStream fileInputStream = new FileInputStream("d:\\tmp\\file01.txt");
        FileChannel fileChannel01 = fileInputStream.getChannel();
        // 输出流
        FileOutputStream fileOutputStream = new FileOutputStream("d:\\tmp\\file02.txt");
        FileChannel fileChannel02 = fileOutputStream.getChannel();

        // 一个Buffer，读写共用
        ByteBuffer byteBuffer = ByteBuffer.allocate(512);

        //数据读写的流程
        while (true) {
            //清空buffer
            byteBuffer.clear();
            // 读
            int read = fileChannel01.read(byteBuffer);
            System.out.println("read =" + read);
            //表示读完
            if (read == -1) {
                break;
            }

            //将buffer 中的数据写入到 fileChannel02 -- 2.txt
            byteBuffer.flip();
            fileChannel02.write(byteBuffer);
        }

        //关闭相关的流
        fileInputStream.close();
        fileOutputStream.close();
    }
}
```

##### Socketchannel

SocketChannel 可以理解成一个 TCP 客户端（其实SocketChannel还可以作为服务端中Worker线程组要处理的TCP长连接），打开一个 TCP 连接的姿势如下：

```java
// 打开一个通道
SocketChannel socketChannel = SocketChannel.open();
// 发起连接
socketChannel.connect(new InetSocketAddress("localhost", 80));
```

读写数据的方式也很方便，读时read到缓冲buffer，写时刷出缓冲buffer即可：

```java
// 读取数据
socketChannel.read(buffer);
// 写入数据到网络连接中
while(buffer.hasRemaining()) {
    socketChannel.write(buffer);   
}
```

##### ServerSocketChannel

ServerSocketChannel 可以理解为服务端，ServerSocketChannel 用于监听机器端口，管理从这个端口进来的 TCP 连接。

```java
// 实例化
ServerSocketChannel serverSocketChannel = ServerSocketChannel.open();
// 监听端口
serverSocketChannel.socket().bind(new InetSocketAddress(80));
while (true) {
    // 一旦有一个 TCP 连接进来，就对应创建一个 SocketChannel 进行处理
    SocketChannel socketChannel = serverSocketChannel.accept();
}
```

ServerSocketChannel 不和 Buffer 打交道了，因为它并不实际处理数据，它一旦接收到请求后，实例化 SocketChannel，之后在这个连接通道上的数据传递它就不管了，因为它需要继续监听端口，等待下一个连接；每一个TCP连接都分配给一个SocketChannel来处理了，读写都基于后面的SocketChannel，这部分其实也是网络编程中经典的Reactor设计模式。

#### Selector

<img src="/img/Netty/NIO-Selector.png" alt="img" style="zoom:50%;" />

Selector是三大组件中的最C位的对象，Selector建立在非阻塞的基础之上，IO多路复用在Java中实现就是它，**它做到了一个线程管理多个Channel**，可以向Selector注册感兴趣的事件，当事件就绪，通过`Selector.select()`方法获取注册的事件，进行相应的操作。

- Selector 能够检测多个注册的通道上是否有**事件**发生（注意：多个Channel以事件的方式可以注册到同一个Selector），如果有事件发生，便获取事件然后针对每个事件进行相应的处理。这样就可以只用一个单线程去管理多个通道，也就是管理多个连接和请求。
- 只有在 连接/通道 真正有读写事件发生时，才会进行读写，就大大地减少了系统开销，并且不必为每个连接都创建一个线程，不用去维护多个线程，避免了多线程之间的上下文切换导致的开销。
- 当线程从某客户端 Socket 通道进行读写数据时，若没有数据可用时，该线程可以进行其他任务。
- 线程通常将非阻塞 IO 的空闲时间用于在其他通道上执行 IO 操作，所以单独的线程可以管理多个输入和输出通道。
- 由于读写操作都是非阻塞的，这就可以充分提升 IO 线程的运行效率，避免由于频繁 I/O 阻塞导致的线程挂起。
- 一个 I/O 线程可以并发处理 N 个客户端连接和读写操作，这从根本上解决了传统同步阻塞 I/O 一连接一线程模型，架构的性能、弹性伸缩能力和可靠性都得到了极大的提升。

##### 事件类型

一个服务端程序启动一个Selector，在Netty中一个NioEventLoop对应一个Selector，Netty在解决JDK NIO的epoll空轮询bug时，采用的策略是废弃原来的有问题的Selector，然后重建一个Selector。因此在Reactor的主从反应堆这里，不同的反应堆可以取不同的Selector事件来选择关心，可以用注册的事件有如下四种：

- SelectionKey.OP_CONNECT
- SelectionKey.OP_ACCEPT
- SelectionKey.OP_READ
- SelectionKey.OP_WRITE

主Reactor通常就是设计为关心的OP_ACCEPT，而从Reactor就更关心其余的读写以及连接状态事件。Selector中的select是实现多路复用的关键，这个方法会一直阻塞直到至少一个channel被选择(即该channel注册的事件发生了为止，除非当前线程发生中断或者selector的wakeup方法被调用。

Selector.select方法最终调用的是EPollSelectorImpl的doSelect方法，在深入远吗就会发现其中的subSelector.poll() ，这里是select的核心，由native函数poll0实现。

##### 编码流程

```java
Selector selector = Selector.open();
// 实例化一个服务端的ServerSocketChannel
ServerSocketChannel serverSocketChannel = ServerSocketChannel.open();
//开启非阻塞
serverSocketChannel.configureBlocking(false);
//绑定端口
serverSocketChannel.socket().bind(new InetSocketAddress(1234));
//ServerSocketChannel注册selector，并表示关心连接事件
serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);
//没有socket就绪，select方法会被阻塞一段时间，并返回0
while (selector.select() > 0) {
    //socket就绪则会返回具体的事件类型和数目
    Set<SelectionKey> keys = selector.selectedKeys();
    //遍历事件
    Iterator<SelectionKey> iterator = keys.iterator();
    //根据事件类型，进行不同的处理逻辑；
    while (iterator.hasNext()) {
       SelectionKey key = iterator.next();
       iterator.remove();
       if (key.isAcceptable()) {
       ...
       } else if (key.isReadable() && key.isValid()) {
       ...
       }
       keys.remove(key);
    }
 }
```

#### 三大件工作流程

流程：

- 首先创建`ServerSocketChannel`对象，和真正处理业务的线程池。
- 然后对上述`ServerSocketChannel`对象进行绑定一个对应的端口，并设置为非阻塞。
- 紧接着创建`Selector`对象并打开，然后把这`Selector`对象注册到`ServerSocketChannel`中，并设置好监听的事件，监听 `SelectionKey.OP_ACCEPT`。
- 接着就是`Selector`对象进行死循环监听每一个`Channel`通道的事件，循环执行 `Selector.select()`方法，轮询就绪的`Channel`。
- 从`Selector中`获取所有的`SelectorKey`（这个就可以看成是不同的事件），如果`SelectorKey`是处于 `OP_ACCEPT`状态，说明是新的客户端接入，调用 `ServerSocketChannel.accept`接收新的客户端。
- 然后对这个把这个接受的新客户端的`Channel`通道注册到`ServerSocketChannel`上，并且把之前的`OP_ACCEPT`状态改为`SelectionKey.OP_READ`读取事件状态，并且设置为非阻塞的，然后把当前的这个`SelectorKey`给移除掉，说明这个事件完成了
- 如果第5步的时候过来的事件不是`OP_ACCEPT`状态，那就是`OP_READ`读取数据的事件状态，然后调用读取数据的机制就可以了。

Netty的网络编程风格上要优化了许多，它的工作流程步骤：

- 创建 NIO 线程组 `EventLoopGroup` 和 `ServerBootstrap`。
- 设置 `ServerBootstrap`的属性：线程组、SO_BACKLOG 选项，设置 `NioServerSocketChannel` 为 `Channel`，设置业务处理 `Handler`
- 绑定端口，启动服务器程序。
- 在业务处理 `Handler处理器` 中，读取客户端发送的数据，并给出响应。

## 五、零拷贝

零拷贝(Zero-copy) 技术是指计算机执行操作时，CPU不需要先将数据从某处内存复制到另一个特定区域。这种技术通常用于通过网络传输文件时节省CPU周期和内存带宽。

**零拷贝的好处：**

- **减少甚至完全避免不必要的CPU拷贝**，从而让CPU解脱出来去执行其他的任务；
- **减少内存带宽的占用**；
- 通常零拷贝技术还能够**减少用户空间和操作系统内核空间之间的上下文切换**。

**零拷贝的实现：**

零拷贝实际的实现并没有真正的标准，取决于操作系统如何实现这一点。

**零拷贝完全依赖于操作系统。**操作系统支持，就有；不支持，就没有。

### 传统IO

#### IO中断与DMA

IO中断，需要CPU响应，需要CPU参与，因此效率比较低。

用户进程需要读取磁盘数据，需要CPU中断，发起IO请求，每次的IO中断，都带来CPU的上下文切换。

**因此出现了——DMA。**

**DMA(Direct Memory Access，直接内存存取)** 是所有现代电脑的重要特色，它允许不同速度的硬件装置来沟通，而不需要依赖于CPU 的大量中断负载。
 DMA控制器，接管了数据读写请求，减少CPU的负担。这样一来，CPU能高效工作了。
 现代硬盘基本都支持DMA。

Linux IO流程：

实际因此IO读取，涉及两个过程：
1、DMA等待数据准备好，把磁盘数据读取到操作系统内核缓冲区；
2、用户进程，将内核缓冲区的数据copy到用户空间。
这两个过程，都是阻塞的。

在Java中，通过InputStream从源数据中读取数据流到一个缓冲区里，然后再将它们输入到OutputStream里。

![img](/img/Netty/零拷贝-传统IO.png)

从上图中可以看出，共产生了**四次数据拷贝**，即使使用了`DMA`来处理了与硬件的通讯，CPU仍然需要处理两次数据拷贝，与此同时，在用户态与内核态也发生了多次上下文切换，无疑也加重了CPU负担。

在此过程中，我们没有对文件内容做任何修改，那么在内核空间和用户空间来回拷贝数据无疑就是一种浪费，而零拷贝主要就是为了解决这种低效性。

### Linux零拷贝机制

#### mmap内存映射

DMA加载磁盘数据到kernel buffer后，应用程序缓冲区(application buffers)和内核缓冲区(kernel buffer)进行映射，数据在应用缓冲区和内核缓存区的改变就能省略。

<img src="/img/Netty/零拷贝-mmap内存映射.png" alt="img" style="zoom:50%;" />

mmap内存映射将会经历：**3次拷贝（1次cpu copy，2次DMA copy）； 以及4次上下文切换。**

#### sendfile

当调用sendfile()时，DMA将磁盘数据复制到kernel buffer，然后将内核中的kernel buffer直接拷贝到socket buffer； 一旦数据全都拷贝到socket buffer，sendfile()系统调用将会return、代表数据转化的完成。 socket buffer里的数据就能在网络传输了。

<img src="/img/Netty/零拷贝-sendfile.png" alt="img" style="zoom:50%;" />

sendfile会经历：**3次拷贝（1次CPU copy 2次DMA copy）； 以及2次上下文切换。**

#### Sendfile With DMA Scatter/Gather Copy

Scatter/Gather可以看作是sendfile的增强版，批量sendfile。

<img src="/img/Netty/零拷贝-sendfile with DBM Scatter Gather.png" alt="img" style="zoom:50%;" />

Scatter/Gather会经历：**2次拷贝（0次cpu copy，2次DMA copy）；以及2次上下文切换。**

#### splice

数据从磁盘读取到OS内核缓冲区后，在内核缓冲区直接可将其转成内核空间其他数据buffer，而不需要拷贝到用户空间。 如下图所示，从磁盘读取到内核buffer后，在内核空间直接与socket buffer建立pipe管道。 和sendfile()不同的是，splice()不需要硬件支持。

<img src="/img/Netty/零拷贝-splice.png" alt="img" style="zoom:50%;" />

注意splice和sendfile的不同，sendfile是将磁盘数据加载到kernel buffer后，需要一次CPU copy,拷贝到socket buffer。 而splice是更进一步，连这个CPU copy也不需要了，直接将两个内核空间的buffer进行set up pipe。

splice会经历：**2次拷贝（0次cpu copy 2次DMA copy）； 以及2次上下文切换。**

#### 对比

| 方法                             | CPU拷贝 | DMA拷贝 | 系统调用   | 上下文切换 |
| -------------------------------- | ------- | ------- | ---------- | ---------- |
| 传统IO                           | 2       | 2       | read/write | 4          |
| mmap                             | 1       | 2       | mmap/write | 4          |
| sendfile                         | 1       | 2       | sendfile   | 2          |
| sendfile with dma scatter/gather | 0       | 2       | sendfile   | 2          |
| splice                           | 0       | 2       | splice     | 2          |

### Java零拷贝机制

Linux提供的领拷贝技术 Java并不是全支持，支持2种(内存映射mmap、sendfile)。

#### NIO提供的内存映射 MappedByteBuffer

首先要说明的是，JavaNlO中 的**Channel (通道)**就相当于操作系统中的**内核缓冲区**，有可能是读缓冲区，也有可能是网络缓冲区，而**Buffer**就相当于操作系统中的**用户缓冲区**。

- Channel  --  内核缓冲区
- Buffer  --  用户缓冲区

**底层就是调用Linux mmap()实现的。**

```java
MappedByteBuffer mappedByteBuffer = new RandomAccessFile(file, "r")
    .getChannel()
    .map(FileChannel.MapMode.READ_ONLY, 0, len);
```

将内核缓冲区的内存和用户缓冲区的内存做了一个地址映射。这种方式**适合读取大文件**，同时也**能对文件内容进行更改**，但是**如果其后要通过SocketChannel发送，还是需要CPU进行数据的拷贝**。 使用MappedByteBuffer，小文件，效率不高；一个进程访问，效率也不高。

MappedByteBuffer只能通过调用FileChannel的map()取得，再没有其他方式。 FileChannel.map()是抽象方法，具体实现是在 FileChannelImpl.c 可查看JDK源码，其map0()方法就是调用了Linux内核的mmap的API。

使用 MappedByteBuffer类要注意的是：**mmap的文件映射，在full gc时才会进行释放。当close时，需要手动清除内存映射文件，可以反射调用sun.misc.Cleaner方法。**

#### NIO提供的sendfile

- **FileChannel.transferTo()方法**直接将当前通道内容传输到另一个通道，没有涉及到Buffer的任何操作，NIO中 的Buffer是JVM堆或者堆外内存，但不论如何他们都是操作系统内核空间的内存
- transferTo()的实现方式就是通过系统调用sendfile() (当然这是Linux中的系统调用)

```java
//使用sendfile:读取磁盘文件，并网络发送
FileChannel sourceChannel = new RandomAccessFile(source, "rw").getChannel();
SocketChannel socketChannel = SocketChannel.open(sa);
sourceChannel.transferTo(0, sourceChannel.size(), socketChannel);
```

ZeroCopyFile实现文件复制：

```java
class ZeroCopyFile {

    public void copyFile(File src, File dest) {
        try (FileChannel srcChannel = new FileInputStream(src).getChannel();
             FileChannel destChannel = new FileInputStream(dest).getChannel()) {

            srcChannel.transferTo(0, srcChannel.size(), destChannel);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

注意： Java NIO提供的FileChannel.transferTo 和 transferFrom 并不保证一定能使用零拷贝。实际上是否能使用零拷贝与操作系统相关，如果操作系统提供 sendfile 这样的零拷贝系统调用，则这两个方法会通过这样的系统调用充分利用零拷贝的优势，否则并不能通过这两个方法本身实现零拷贝。



## 六、Netty

### 简介

> Netty is *an asynchronous event-driven network application framework*
> for rapid development of maintainable high performance protocol servers & clients.

简介：

- Netty 是由 JBOSS 提供的一个 Java 开源框架。Netty 提供异步的、基于事件驱动的网络应用程序框架，用以快速开发高性能、高可靠性的网络 IO 程序
- Netty 可以帮助你快速、简单的开发出一个网络应用，相当于简化和流程化了 NIO 的开发过程
- Netty 是目前最流行的 NIO 框架，Netty 在互联网领域、大数据分布式计算领域、游戏行业、通信行业等获得了广泛的应用，知名的 Elasticsearch 、Dubbo 框架内部都采用了 Netty。

优点：

- 设计优雅：适用于各种传输类型的统一 API 阻塞和非阻塞 Socket；基于灵活且可扩展的事件模型，可以清晰地分离关注点；高度可定制的线程模型 - 单线程，一个或多个线程池。
- 使用方便：详细记录的 Javadoc，用户指南和示例；没有其他依赖项，JDK 5（Netty 3.x）或 6（Netty 4.x）就足够了。
- 高性能、吞吐量更高：延迟更低；减少资源消耗；最小化不必要的内存复制。

- 安全：完整的 SSL/TLS 和 StartTLS 支持。

### Netty模型

工作原理示意图：

 <img src="/img/Netty/Netty Reactor工作架构图.png" alt="img" style="zoom:80%;" />

 

- Netty抽象出**两组线程池**：
  - **BossGroup**：专门负责接收客户端的连接；
  - **WorkerGroup**： 专门负责网络的读写；
  - BossGroup 和 WorkerGroup 类型都是 **NioEventLoopGroup**。

- NioEventLoopGroup 相当于一个**事件循环组**, 这个组中含有多个事件循环 ，每一个事件循环是 NioEventLoop。
- **NioEventLoop** 表示一个不断循环的执行处理任务的线程， **每个NioEventLoop 都有一个selector** , 用于监听绑定在其上的socket的网络通讯。
- NioEventLoopGroup 可以有多个线程, 即可以含有多个NioEventLoop。
- 每个**Boss NioEventLoop** 循环执行的步骤有3步：
  - 轮询accept 事件；
  - 处理accept 事件 , 与client建立连接 , 生成NioScocketChannel , 并将其注册到某个worker NIOEventLoop 上的 selector；
  - 处理任务队列的任务 ， 即 runAllTasks。
- 每个 **Worker NIOEventLoop** 循环执行的步骤：
  - 轮询read, write 事件；
  - 处理i/o事件， 即read , write 事件，在对应NioScocketChannel 处理；
  - 处理任务队列的任务 ， 即 runAllTasks。
- 每个Worker NIOEventLoop  处理业务时，会使用**pipeline（管道）**, **pipeline 中包含了 channel** , 即通过pipeline 可以获取到对应通道, 管道中维护了很多的 处理器。










## References

- [小六六学Netty系列之Java BIO](https://juejin.cn/post/6859336784627646471)
- [小六六学Netty系列之unix IO模型](https://juejin.cn/post/6862141145553338382/)
- [小六六学Netty系列之Java NIO(一)](https://juejin.cn/post/6860275544655659015/)
- [小六六学Netty系列之Java NIO(二)](https://juejin.cn/post/6860798575187197965)
- [这次答应我，一举拿下 I/O 多路复用！](https://mp.weixin.qq.com/s/Qpa0qXxuIM8jrBqDaXmVNA)
- [一文搞定NIO的三大组件](https://www.jianshu.com/p/0bb17a7a63a0)
- [浅析Linux中的零拷贝技术](https://www.jianshu.com/p/fad3339e3448)
- [Java中的零拷贝](https://www.jianshu.com/p/2fd2f03b4cc3)
- [操作系统之 高性能IO浅析](https://icefrozen.github.io/article/operating-system-io/)



