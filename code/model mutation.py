#进行高斯模糊
def GF_mut(model, mutation_ratio, distribution='normal', STD=0.1, lower_bound=None, upper_bound=None):

    # 正态分布和均匀分布 这里是选则正态分布，如果是均匀分布就报错
    valid_distributions = ['normal', 'uniform']
    assert distribution in valid_distributions, 'Distribution %s is not support.' % distribution
    if distribution == 'uniform' and (lower_bound is None or upper_bound is None):
        mylogger.error('Lower bound and Upper bound is required for uniform distribution.')
        raise ValueError('Lower bound and Upper bound is required for uniform distribution.')   
    #进行模型复制
    mylogger.info('copying model...')
    GF_model = utils.ModelUtils.model_copy(model, 'GF')
    mylogger.info('model copied')
    chosed_index = np.random.randint(0, len(GF_model.layers))
    # 从模型中获取所有的层
    layer = GF_model.layers[chosed_index]
    mylogger.info('executing mutation of {}'.format(layer.name))
    #获取每一层的权重
    weights = layer.get_weights()
    new_weights = []
    # 执行高斯模糊 对每一个权重进行正态处理
    for weight in weights:
        weight_shape = weight.shape
        weight_flat = weight.flatten()
        permu_num = math.floor(len(weight_flat) * mutation_ratio)
        permutation = np.random.permutation(len(weight_flat))[:permu_num]
        STD = math.sqrt(weight_flat.var()) * STD
        # 正太化
        weight_flat[permutation] += np.random.normal(scale=STD, size=len(permutation))
        weight = weight_flat.reshape(weight_shape)
        new_weights.append(weight)
    layer.set_weights(new_weights)
    # 返回处理后的高斯模型模型
    return GF_model


# 交换神经元之间链接的权重
def WS_mut(model, mutation_ratio, mutated_layer_indices=None):
    # 首先进行模型复制
    WS_model = utils.ModelUtils.model_copy(model, 'WS')
    # 获取模型中所有的层
    layers = WS_model.layers
    # 求出layers的总个数
    depth_layer = len(layers)
    mutated_layer_indices = np.arange(depth_layer) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, depth_layer)
        np.random.shuffle(mutated_layer_indices)
        i = mutated_layer_indices[0]
        layer = layers[i]
        weights = layer.get_weights()
        layer_name = type(layer).__name__
        # 判断layer层是Conv2D还是Dense
        if layer_name == "Conv2D" and len(weights) != 0:
        # 根据不同的结构，交换层级之间的神经元权重
            layer.set_weights(_shuffle_conv2d(weights, mutation_ratio))
        elif layer_name == "Dense" and len(weights) != 0:
            layer.set_weights(_shuffle_dense(weights, mutation_ratio))
        else:
            pass
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")
    return WS_model

#将该神经元设置到下一层的权重设置为0
def NEB_mut(model, mutation_ratio, mutated_layer_indices=None):
    # 先进行复制
    NEB_model = utils.ModelUtils.model_copy(model, 'NEB')
    layers = NEB_model.layers
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, len(layers))
        layer_utils = LayerUtils()
        # 将indices 进行打乱
        np.random.shuffle(mutated_layer_indices)
        # 遍历当中的每一个 indices
        for i in mutated_layer_indices:
            layer = layers[i]
            #如果权重不在当前的list中 ，则跳过
            if not layer_utils.is_layer_in_weight_change_white_list(layer):
                continue

            weights = layer.get_weights()
            if len(weights) > 0:
                if isinstance(weights, list):
                    # assert len(weights) == 2
                    if len(weights) != 2:
                        continue
                    else:
                        weights_w, weights_b = weights
                        weights_w = weights_w.transpose()
                        permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                        # 用于存放下层的每个神经元的值
                        weights_w[permutation] = np.zeros(weights_w[0].shape)
                        weights_w = weights_w.transpose()
                        weights_b[permutation] = 0
                        # weights已经是一个全为0的数组
                        weights = weights_w, weights_b
                        # 将下一层的权重全部设置为0
                        layer.set_weights(weights)
                else:
                    # 如果不是一个实例，则将他转换为一个实例
                    assert isinstance(weights, np.ndarray)
                    weights_w = weights
                    weights_w = weights_w.transpose()
                    permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                    weights_w[permutation] = np.zeros(weights_w[0].shape)
                    weights_w = weights_w.transpose()
                    # 产生一个为0的数组 为下一层赋值
                    weights = [weights_w]
                    layer.set_weights(weights)
                break
        return NEB_model
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")

# 在将神经元传递给激活函数之前，改变一个神经元的输出值的符号
def NAI_mut(model, mutation_ratio, mutated_layer_indices=None):
    # 进行相似的操作 复制 获得变异层的索引
    NAI_model = utils.ModelUtils.model_copy(model, 'NAI')
    layers = NAI_model.layers
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices   
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, len(layers))
        # 将索引层随机打乱
        np.random.shuffle(mutated_layer_indices)
        layer_utils = LayerUtils()
        for i in mutated_layer_indices:
            layer = layers[i]
            if not layer_utils.is_layer_in_weight_change_white_list(layer):
                continue
            # 获取该层的权重
            weights = layer.get_weights()
            # 这边都与之前的类似
            if len(weights) > 0:
                if isinstance(weights, list):
                    if len(weights) != 2:
                        continue
                    else:
                        weights_w, weights_b = weights
                        weights_w = weights_w.transpose()
                        permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                        # print(permutation)
                        # 将permutation所有的值*-1 即实现了改变改神经元的输出符号
                        weights_w[permutation] *= -1
                        weights_w = weights_w.transpose()
                        weights_b[permutation] *= -1
                        weights = weights_w, weights_b
                        layer.set_weights(weights)
                else:
                    weights_w = weights[0]
                    weights_w = weights_w.transpose()
                    # 产生一个变异体
                    permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                    # print(permutation)
                    weights_w[permutation] *= -1
                    weights_w = weights_w.transpose()
                    weights = [weights_w]
                    layer.set_weights(weights)
                break
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")
    return NAI_model

# 交换两个神经元及其链接的位置
def NS_mut(model, mutated_layer_indices=None):
    NS_model = utils.ModelUtils.model_copy(model, 'NS')
    layers = NS_model.layers
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(mutated_layer_indices, len(layers))
    layer_utils = LayerUtils()
    for i in mutated_layer_indices:
        layer = layers[i]
        if not layer_utils.is_layer_in_weight_change_white_list(layer):
            continue
        weights = layer.get_weights()
        # 与前面都一样 获取每个layer的权重
        if len(weights) > 0:
            # 判断它是不是一个实例 如果是的话就执行NS，如果不是，就把他转换为一个实例
            if isinstance(weights, list):
                if len(weights) != 2:
                    continue
                weights_w, weights_b = weights
                weights_w = weights_w.transpose()
                # 神经元的个数一个大于2 否则无法进行交换
                if weights_w.shape[0] >= 2:
                    # 将permutat[0]与permutation[1]进行交换
                    permutation = np.random.permutation(weights_w.shape[0])[:2]
                    weights_w[permutation[0]], weights_w[permutation[1]] = \
                        weights_w[permutation[1]].copy(), weights_w[permutation[0]].copy()
                    weights_w = weights_w.transpose()
                    weights_b[permutation[0]], weights_b[permutation[1]] = \
                        weights_b[permutation[1]].copy(), weights_b[permutation[0]].copy()
                    weights = weights_w, weights_b
                    # 完成交换 进行赋值
                    layer.set_weights(weights)
                else:
                    mylogger.warning("NS not used! One neuron can't be shuffle!")
            else:
                assert isinstance(weights, np.ndarray)
                # 转化为实例后 执行上面的操作
                weights_w = weights
                weights_w = weights_w.transpose()
                if weights_w.shape[0] >= 2:
                    permutation = np.random.permutation(weights_w.shape[0])[:2]

                    weights_w[permutation[0]], weights_w[permutation[1]] = \
                        weights_w[permutation[1]].copy(), weights_w[permutation[0]].copy()
                    weights_w = weights_w.transpose()
                    weights = [weights_w]

                    layer.set_weights(weights)
                else:
                    mylogger.warning("NS not used! One neuron can't be shuffle!")
            break

    return NS_model

# 移除激活函数
def ARem_mut(model, mutated_layer_indices=None):
    ARem_model = utils.ModelUtils.model_copy(model, 'ARem')
    layers = ARem_model.layers
    # 最后一层的激活函数无法被移除
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    np.random.shuffle(mutated_layer_indices)
    _assert_indices(mutated_layer_indices, len(layers))

    for i in mutated_layer_indices:
        layer = layers[i]
        # 将该层的函数设置为no_actication 这样就无法被使用
        if hasattr(layer, 'activation') and 'softmax' not in layer.activation.__name__.lower():
            layer.activation = ActivationUtils.no_activation
            break
    return ARem_model

# 替换激活函数
def ARep_mut(model, new_activations=None, mutated_layer_indices=None):

    activation_utils = ActivationUtils()
    ARep_model = utils.ModelUtils.model_copy(model, 'ARep')
    layers = ARep_model.layers
    # 最后一层的激活函数无法被替换
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    # 对变异层索引进行随机打乱
    np.random.shuffle(mutated_layer_indices)
    _assert_indices(mutated_layer_indices, len(layers))
    for i in mutated_layer_indices:
        layer = layers[i]
        # 如果该层是处于被激活的状态 ，则随意选择一个激活函数将替换
        if hasattr(layer, 'activation') and 'softmax' not in layer.activation.__name__.lower():
            layer.activation = activation_utils.pick_activation_randomly(new_activations)
            break
    return ARep_model

# 为模型增加一个新的层
def LA_mut(model, new_layers=None, mutated_layer_indices=None):
    layer_utils = LayerUtils()
    # 先判断这个新的层在不在原来的层里面
    if new_layers is not None:
        for layer in new_layers:
            if layer not in layer_utils.available_model_level_layers.keys():
                mylogger.error('Layer {} is not supported.'.format(layer))
                raise Exception('Layer {} is not supported.'.format(layer))
    LA_model = utils.ModelUtils.model_copy(model, 'LA')
    # 遍历所有的层 查看有没有符合插入的层
    insertion_points = _LA_model_scan(LA_model, new_layers, mutated_layer_indices)
    if len(insertion_points.keys()) == 0:
        mylogger.warning('no appropriate layer to insert')
        return None
    for key in insertion_points.keys():
        mylogger.info('{} can be added after layer {} ({})'
            .format(insertion_points[key], key, type(model.layers[key])))
    # 找到可插入层的信息
    layers_index_avaliable = list(insertion_points.keys())
    layer_index_to_insert = layers_index_avaliable[np.random.randint(0, len(layers_index_avaliable))]
    available_new_layers = insertion_points[layer_index_to_insert]
    layer_name_to_insert = available_new_layers[np.random.randint(0, len(available_new_layers))]
    mylogger.info('insert {} after {}'.format(layer_name_to_insert, LA_model.layers[layer_index_to_insert].name))
    # 插入一个新的层
    if model.__class__.__name__ == 'Sequential':
        import keras
        # 新建一个Sequential model
        new_model = keras.models.Sequential()
        for i, layer in enumerate(LA_model.layers):
            new_layer = LayerUtils.clone(layer)
            new_model.add(new_layer)
            # 将新的层插入到该层的后面去
            if i == layer_index_to_insert:
                output_shape = layer.output_shape
                new_model.add(layer_utils.available_model_level_layers[layer_name_to_insert](output_shape))
    else:

        def layer_addition(x, layer):
            x = layer(x)
            output_shape = layer.output_shape
            new_layer = layer_utils.available_model_level_layers[layer_name_to_insert](output_shape)
            x = new_layer(x)
            return x
        new_model = utils.ModelUtils.functional_model_operation(LA_model, operation={LA_model.layers[layer_index_to_insert].name: layer_addition})
    #将模型总的层数加1 
    assert len(new_model.layers) == len(model.layers) + 1
    tuples = []
    import time
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer
    # 更新旧的层级信息
    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name

        if layer_name.endswith('_copy_LA'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()
        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            for i in range(len(shape_sw)):
                assert shape_sw[i] == shape_w[i], '{}'.format(layer_name)
            tuples.append((sw, w))
    # 返回跟新完的新的model
    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model

# 为模型增加多个层
def MLA_mut(model, new_layers = None, mutated_layer_indices=None):
    # mutiple layers addition
    layer_matching = LayerMatching()
    # 与上面算法一样，先判断有没有适合的层进行插入
    if new_layers is not None:
        for layer in new_layers:
            if layer not in layer_matching.layer_concats.keys():
                raise Exception('Layer {} is not supported.'.format(layer))
    MLA_model = utils.ModelUtils.model_copy(model, 'MLA')
    insertion_points = _MLA_model_scan(model, new_layers, mutated_layer_indices)
    mylogger.info(insertion_points)
    if len(insertion_points.keys()) == 0:
        mylogger.warning('no appropriate layer to insert')
        return None
    # 如果有 就找出能够插入的位置
    for key in insertion_points.keys():
        mylogger.info('{} can be added after layer {} ({})'
                             .format(insertion_points[key], key, type(model.layers[key])))

    # 随机选择一个新的层 将其加入到能够被插入的层的后面
    layers_index_avaliable = list(insertion_points.keys())
    layer_index_to_insert = layers_index_avaliable[np.random.randint(0, len(layers_index_avaliable))]
    available_new_layers = insertion_points[layer_index_to_insert]
    layer_name_to_insert = available_new_layers[np.random.randint(0, len(available_new_layers))]
    mylogger.info('choose to insert {} after {}'.format(layer_name_to_insert, MLA_model.layers[layer_index_to_insert].name))
    # 为模型插入多个层级
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        # 这里用for循环来新建多个层
        for i, layer in enumerate(MLA_model.layers):
            new_layer = LayerUtils.clone(layer)
            # new_layer.name += "_copy"
            new_model.add(new_layer)
            # 完成新建 下面执行插入操作
            if i == layer_index_to_insert:
                output_shape = layer.output.shape.as_list()
                layers_to_insert = layer_matching.layer_concats[layer_name_to_insert](output_shape)
                for layer_to_insert in layers_to_insert:
                    layer_to_insert.name += "_insert"
                    mylogger.info(layer_to_insert)
                    new_model.add(layer_to_insert)
        new_model.build(MLA_model.input_shape)
    else:
        def layer_addition(x, layer):
            x = layer(x)
            output_shape = layer.output.shape.as_list()
            new_layers = layer_matching.layer_concats[layer_name_to_insert](output_shape)
            for l in new_layers:
                l.name += "_insert"
                mylogger.info('insert layer {}'.format(str(l)))
                x = l(x)
            return x
        new_model = utils.ModelUtils.functional_model_operation(MLA_model, operation={MLA_model.layers[layer_index_to_insert].name: layer_addition})

    tuples = []
    import time
    start_time = time.time()
    # 更新model的信息
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_MLA'):
            key = layer_name[:-9]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))
    # 计算该步骤的运行时间
    import keras.backend as K
    K.batch_set_value(tuples)
    end_time = time.time()
    print('set weight cost {}'.format(end_time - start_time))

    return new_model

# 将模型中的某一层进行复制
def LC_mut(model, mutated_layer_indices=None):
    # 复制模型
    LC_model = utils.ModelUtils.model_copy(model, 'LC')
    available_layer_indices = _LC_and_LR_scan(LC_model, mutated_layer_indices)

    if len(available_layer_indices) == 0:
        mylogger.warning('no appropriate layer to copy (input and output shape should be same)')
        return None

    # 复制最后一个可用的层级  -1 就是从最后一个开始寻找
    copy_layer_index = available_layer_indices[-1]
    # 找到了最后一个可用的层级
    copy_layer_name = LC_model.layers[copy_layer_index].name + '_repeat'

    mylogger.info('choose to copy layer {}'.format(LC_model.layers[copy_layer_index].name))

    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(LC_model.layers):
            # 将该层集复制并加入到模型当中去
            new_model.add(LayerUtils.clone(layer))
            if i == copy_layer_index:
                copy_layer = LayerUtils.clone(layer)
                copy_layer.name += '_repeat'
                new_model.add(copy_layer)
    else:
        def layer_repeat(x, layer):
            x = layer(x)
            copy_layer = LayerUtils.clone(layer)
            copy_layer.name += '_repeat'
            x = copy_layer(x)
            return x
        new_model = utils.ModelUtils.functional_model_operation(LC_model, operation={LC_model.layers[copy_layer_index].name: layer_repeat})

    # 重新更新层级之间的权重
    assert len(new_model.layers) == len(model.layers) + 1
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer
    # 更新layers的名称
    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LC'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer
    #更新 layers的键值
    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        if layer_name + '_copy_LC_repeat' == copy_layer_name:
            for sw, w in zip(new_model_layers[copy_layer_name].weights, layer_weights):
                shape_sw = np.shape(sw)
                shape_w = np.shape(w)
                assert len(shape_sw) == len(shape_w)
                assert shape_sw[0] == shape_w[0]
                tuples.append((sw, w))
    # 更新数组大小
        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))
    # 返回更新完的新模型
    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model

# 从模型中移除一层
def LR_mut(model, mutated_layer_indices=None):
    # 获取模型 得到层级索引
    LR_model = utils.ModelUtils.model_copy(model, 'LR')
    available_layer_indices = _LC_and_LR_scan(LR_model, mutated_layer_indices)
    # 判断有没有可以移除的层级
    if len(available_layer_indices) == 0:
        mylogger.warning('no appropriate layer to remove (input and output shape should be same)')
        return None
    # 移除最近的一个可移除的层级 -1 就是从最后开始寻找
    remove_layer_index = available_layer_indices[-1]
    mylogger.info('choose to remove layer {}'.format(LR_model.layers[remove_layer_index].name))
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        # 对所有层级的index进行遍历 如果该层级的index不是需要删除层级的index 则把它加入到新的层级当中去
        for i, layer in enumerate(LR_model.layers):
            if i != remove_layer_index:
                new_layer = LayerUtils.clone(layer)
                new_model.add(new_layer)
    else:
        new_model = utils.ModelUtils.functional_model_operation(LR_model, operation={LR_model.layers[remove_layer_index].name: lambda x, layer: x})

   # 更新权重
    assert len(new_model.layers) == len(model.layers) - 1
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer
    # 修改新模型的名称
    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LR'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer
    # 修改新模型的键值和权
    for layer_name in new_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))
    # 返回新的模型
    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model

# 交换模型中的两个层级
def LS_mut(model):
    # 先将模型进行复制
    LS_model = utils.ModelUtils.model_copy(model,"LS")
    shape_dict = _LS_scan(LS_model)
    layers = LS_model.layers
    # 创建一个交换列表
    swap_list = []
    # 将model中符合条件的层架加入到交换列表的中
    for v in shape_dict.values():
        if len(v) > 1:
            swap_list.append(v)
    if len(swap_list) == 0:
        mylogger.warning("No layers to swap!")
        return None
    #随机从列表中选出两个层级
    swap_list = swap_list[random.randint(0, len(swap_list)-1)]
    choose_index = random.sample(swap_list, 2)
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.Sequential()
        # 将layer1和layer进行交换
        for i, layer in enumerate(layers):
            if i == choose_index[0]:
                new_model.add(LayerUtils.clone(layers[choose_index[1]]))
            elif i == choose_index[1]:
                new_model.add(LayerUtils.clone(layers[choose_index[0]]))
            else:
                new_model.add(LayerUtils.clone(layer))
    else:
        layer_1 = layers[choose_index[0]]
        layer_2 = layers[choose_index[1]]
        new_model = utils.ModelUtils.functional_model_operation(LS_model, {layer_1.name: lambda x, layer: LayerUtils.clone(layer_2)(x),
                                                           layer_2.name: lambda x, layer: LayerUtils.clone(layer_1)(x)})

    # 更新权重
    assert len(new_model.layers) == len(model.layers)
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer
    # 设置新模型中层级的名称
    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LS'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer
    # 设置新模型中层级的键值
    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))
    #返回新的模型
    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model
