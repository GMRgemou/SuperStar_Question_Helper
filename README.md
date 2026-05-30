# 学习通答题助手
旨在帮助更多大学生脱离琐事烦恼，享受大学生活

实现方式参考 SuperStar 项目，复用其：
1. 登录 / Session / cookies 逻辑
2. 课程列表 / 章节 / 任务点解析逻辑
3. 题目页面解析逻辑
4. 提交答题表单的字段组织方式

支持两种入口：
1. 直接传作业/测验 URL
2. 登录后传课程码，自动遍历课程中的答题任务

答题来源支持两种：
1. 本地题库 JSON
2. OpenAI 兼容接口

食用方法：
0. apikey从哪来？去各大模型网站注册，默认是硅基流动
1. 配置chaoxing_quiz_agent.local，不建议替换base_url，因为默认模型是免费的
2. python chaoxing_quiz_agent.py 课题号 --mode course --username 用户名 --password 密码
3. 如果您在.py文件替换了模型，请确保base_url和api_key同步正确更换

特别鸣谢
感谢https://github.com/lispringing/SuperStar
感谢提供的部分思路
