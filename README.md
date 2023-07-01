# For TCCI base learning
TCCI初级教程，每一周的文档放在对应的文件夹中。

# CHATGPT相关工具使用说明准备上线
CHATPDF https://www.chatpdf.com 基础PDF阅读，泛读推荐

ELICIT  https://elicit.org/ 相关文献搜索，模糊

EXPLAINPAPER https://www.explainpaper.com 偏重内容理解，精读推荐

# WEEK1.
Git的常用命令和基本语法，PPT加练习
## 第一周和第三周涉及文献和视频资料

generative agent：https://arxiv.org/pdf/2304.03442.pdf

demo地址：https://reverie.herokuapp.com/arXiv_Demo/

李宏毅讲解视频：https://youtu.be/G44Lkj7XDsA

# WEEK2.
BERT微调训练，以AGNEWS为例进行训练，并完成该项任务。
微调代码已上传，agnews-attack.py对BERT进行了微调，并保存调好的模型。
bert-snil.py是微调好的snil任务。
chat_attack.py是使用chat进行prompt工程的范例。

unittest的使用，测试概念
相关资料已经上传至UnitTest文件夹中

BERT原文：https://arxiv.org/abs/1810.04805

huggingface：https://arxiv.org/pdf/1910.03771.pdf

# WEEK3.

强化学习概念，以HUGGING FACE博客为例，训练出一个成功的AGENT
huggingface blog：https://huggingface.co/learn/deep-rl-course/en/unit1/hands-on?fw=pt

RL结合CHAT进行开发的思路（part1-讲解基础的强化学习知识，训练出一个Agent并知道如何自定义一个环境
NLPGym: https://github.com/rajcscw/nlp-gym
Overview: https://www.ijcai.org/proceedings/2019/0880.pdf

TODO..
part2-知道强化学习在NLP中的应用方式，并把相关任务用强化学习进行优化
创新点1. 强化学习作为instruction输出，创造一个生态链，仿造一个群体算法进行优化
创新点2. 强化学习对对抗样本进行防御，尝试让对抗样本对模型的影响降低
创新点3. 多个模型进行交互...

# WEEK4.

对抗训练，对抗样本
...
