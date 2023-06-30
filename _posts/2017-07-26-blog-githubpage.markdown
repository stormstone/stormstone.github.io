---
layout:     post
title:      "立马快速打造自己的个人博客"
date:       2017-07-26 21:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
---



## First

管他的，先引用一段：<br>

“喜欢写Blog的人，会经历三个阶段。<br>

　　第一阶段，刚接触Blog，觉得很新鲜，试着选择一个免费空间来写。

　　第二阶段，发现免费空间限制太多，就自己购买域名和空间，搭建独立博客。

　　第三阶段，觉得独立博客的管理太麻烦，最好在保留控制权的前提下，让别人来管，自己只负责写文章。

​	大多数Blog作者，都停留在第一和第二阶段，因为第三阶段不太容易到达：你很难找到俯首听命、愿意为你管理服务器的人。


　　但是两年前，情况出现变化，一些程序员开始在ｇithub网站上搭建blog。他们既拥有绝对管理权，又享受github带来的便利----不管何时何地，只要向主机提交commit，就能发布新文章。更妙的是，这一切还是免费的，github提供无限流量，世界各地都有理想的访问速度。
　　
　　今天，我就来示范如何在github上搭建Blog，你可以从中掌握github的Pages功能，以及Jekyll软件的基本用法。更重要的是，你会体会到一种建立网站的全新思路。“

## Second

​	的确，这是我们很多人对Blog的经历。现在就来看看第三阶段，我们用GitHub pages轻松实现。

真的很简单：


1.在GitHub创建项目：<br>

	Create a new repository named username.github.io, where username is your username (or organization name) on GitHub.


（经过亲身测试，只能是自己GitHub的用户名或组织名，用其他名字无效！）

2.从GitHub克隆到本地：<br>

	~$ git clone https://github.com/username/username.github.io


3.Hello World 是所有程序开始的标配：<br>

	~$ echo "Hello World" > index.html
blog首页默认为index.html。

4.Push it，Add, commit, and push your changes，推送到GitHub：<br>

	~$ git add --all
	~$ git commit -m "Initial commit"
	~$ git push -u origin master


就这样访问 https://username.github.io，就可以看到HelloWord了！成功搞定了，是不是so easy！

这个过程官网也是非常清楚:[https://pages.github.com/](https://pages.github.com)。

## Third

之后，整个blog的建设就得靠自己了，和其他网页开发一样，用HTML+CSS+JavaScript实现。根路径也是用“/”表示，要引用直接相对路径就可以了。

还有，可以使用Jekyll对静态网页编写。
jekyll是一个简单的免费的Blog生成工具，类似WordPress。但是和WordPress又有很大的不同，原因是jekyll只是一个生成静态网页的工具，不需要数据库支持。但是可以配合第三方服务,例如Disqus。最关键的是jekyll可以免费部署在Github上，而且可以绑定自己的域名（[百度百科](https://baike.baidu.com/item/jekyll/1164861?fr=aladdin)）。

jekyll官网：[http://jekyll.com.cn/](http://jekyll.com.cn/)

其实也不用自己去写，直接去GitHub找几个好的模板，修改修改，就可以了，呵呵呵！(σﾟ∀ﾟ)σ..:*☆哎哟不错哦。

我的博客[SH Blog](https://stormstone.github.io/) copy from[Hux Blog](http://huangxuan.me/)。
github地址：[http://github.com/Huxpro/huxpro.github.io](http://github.com/Huxpro/huxpro.github.io)。

至于项目里的各个文件的含义，jekyll语法，就自己慢慢摸索咯，我也不造啊！




**以上**