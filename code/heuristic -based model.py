import random

rules = []  # 将所以的规则放在一个列表里面
seeds = []  # 将所有的模型放在一个列表里面
models = []  # 将所有的结果模型放在一个列表里面
pro = {}  # 每个模型出现的概率 格式为字典 model:概率
times = {}  # 每个模型出现的次数 格式为字典 model：次数
position = []  # 用来记录每个rule的排名
p=0.3


# 由模型出现的次数计算的它的得分
def cal_score(model):
    return 1 / times[model] + 1


# 根据模型的得分计算出该seeds被选中的概率
def cal_prob(model):
    cnt = 0
    for i in range(len(seeds)):
        cnt = cnt + cal_score(seeds[i])
    return cal_score(model) / cnt


# 模型+规则进行变异操作
def mutate(model,rule):
    return model



# 启发式算法的调用
def heuristic_alg(rules,seeds,n):
    global k_b, k_a, s
    mu_a = random.choice(rules)
    # 执行Roulette Wheel Selection
    while len(models) < n:  #n 需要的模型数量
        for i in range(len(seeds)):
            pro[seeds[i]] = cal_prob(seeds[i])
        r = random.random()  # 随机（0-1）之间的小数
        bound = 0
        for i in range(len(models)):
            bound = bound + pro[seeds[i]]
            if r < bound:
                s = seeds[i]
                break
        # s是选择出来的seed 下面对rule进行选择
        for i in range(len(position)):
            if position[i] == mu_a:
                k_a = i               # 找到a的排名

        while True:
            mu_b = random.choice(rules)
            for i in range(len(position)):
                if position[i] == mu_b:
                    k_b = i  # 找到b的排名
            f = random.random()  # 随机（0-1）之间的小数
            # 利用概率分布
            if f > pow(1-p, k_b-k_a):
                continue
            else:
                break
        m = mutate(s, mu_b)
        if m not in models:
            models.append(m)
        if acc(m) >= acc(s):
            if m not in seeds:
                seeds.append(m)
        rules.sort() #重新进行排序，进入下一次迭代
        mu_a=mu_b
    return models


        # selects the next mutation rule MUb based on the current one MUa

