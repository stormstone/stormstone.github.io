---
layout:     post
title:      "项目管理工具--Maven 学习笔记"
date:       2017-08-01 17:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - Maven
---

## 什么是Maven

Maven是一个项目管理工具，它包含了一个项目对象模型 (Project Object Model)，一组标准集合，一个项目生命周期(Project Lifecycle)，
一个依赖管理系统(Dependency Management System)，和用来运行定义在生命周期阶段(phase)中插件(plugin)目标(goal)的逻辑。
当你使用Maven的时候，你用一个明确定义的项目对象模型来描述你的项目，然后Maven可以应用横切的逻辑，这些逻辑来自一组共享的（或者自定义的）插件。

## 常用命令编辑
- mvn archetype：generate 创建Maven项目（create过时）
- mvn compile 编译源代码
- mvn deploy 发布项目
- mvn test-compile 编译测试源代码
- mvn test 运行应用程序中的单元测试
- mvn site 生成项目相关信息的网站
- mvn clean 清除项目目录中的生成结果
- mvn package 根据项目生成的jar
- mvn install 在本地Repository中安装jar
- mvn eclipse:eclipse 生成eclipse项目文件
- mvnjetty:run 启动jetty服务
- mvntomcat:run 启动tomcat服务
- mvn clean package -Dmaven.test.skip=true:清除以前的包后重新打包，跳过测试类

## 项目目录结构

	-src
		--main
			---java
				----package
		--test
			---java
				----package
		--resources
	-target
	-pom.xml

- /src/main/java/package:项目的源代码
- /src/test/java/package:项目测试代码
- /src/resources:资源文件
- /target: maven自动生成，包括编译后的class文件，导出的jar包
- /pom.xml: maven配置文件，各种依赖配置

## maven坐标
pom.xml：
```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
     http://maven.apache.org/maven-v4_0_0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.seckill</groupId>
  <artifactId>seckill</artifactId>
  <packaging>war</packaging>
  <version>1.0-SNAPSHOT</version>
  <name>seckill Maven Webapp</name>
  <url>http://maven.apache.org</url>
  <dependencies>
      <dependency>
          <!--使用Junit4（注解方式运行）-->
          <groupId>junit</groupId>  <!-- 模块所属组织 -->
          <artifactId>junit</artifactId>  <!-- 引入模块名称 -->
          <version>4.12</version>  <!-- 版本号 -->
          <scope>test</scope>  <!-- 管理依赖的部署 -->
      </dependency>
    </dependencies>

  <build>
      <finalName>projectname</finalName> <!-- 项目名称 -->
  </build>
</project>
```
其中：<scope> </scope> 标签，它主要管理依赖的部署。目前<scope>可以使用5个值： 

* compile，缺省值，适用于所有阶段，会随着项目一起发布。 
* provided，类似compile，期望JDK、容器或使用者会提供这个依赖。如servlet.jar。 
* runtime，只在运行时使用，如JDBC驱动，适用运行和测试阶段。 
* test，只在测试时使用，用于编译和运行测试代码。不会随项目发布。 
* system，类似provided，需要显式提供包含依赖的jar，Maven不会在Repository中查找它。 


## maven生命周期

- validate
- generate-sources
- process-sources
- generate-resources
- process-resources     复制并处理资源文件，至目标目录，准备打包。
- compile     编译项目的源代码。
- process-classes
- generate-test-sources 
- process-test-sources 
- generate-test-resources
- process-test-resources     复制并处理资源文件，至目标测试目录。
- test-compile     编译测试源代码。
- process-test-classes
- test     使用合适的单元测试框架运行测试。这些测试代码不会被打包或部署。
- prepare-package
- package     接受编译好的代码，打包成可发布的格式，如 JAR 。
- pre-integration-test
- integration-test
- post-integration-test
- verify
- install     将包安装至本地仓库，以让其它项目依赖。
- deploy     将最终的包复制到远程的仓库，以让其它开发人员与项目共享。


## 总结
Maven2.0 有着许多实用的特点，并且完成任务十分出色。Maven中最值得称赞的地方就是使用了标准的目录结构和部署。
这就使得开发人员能够适应不同的项目，并且不用学习任何结构方面新的东西，也不用掌握特殊的指令来构建结构。