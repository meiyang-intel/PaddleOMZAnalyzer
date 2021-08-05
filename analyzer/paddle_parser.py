import paddle
paddle.enable_static()
import paddle.fluid as fluid
import paddle.fluid.core as core

def append_fetch_ops(program, fetch_target_names, fetch_holder_name='fetch'):
    """
    In this palce, we will add the fetch op
    """
    global_block = program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)
    print("the len of fetch_target_names:%d" % (len(fetch_target_names)))
    for i, name in enumerate(fetch_target_names):

        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})
        
def insert_fetch(program, fetchs, fetch_holder_name="fetch"):
    global_block = program.global_block()
    need_to_remove_op_index = list()
    for i, op in enumerate(global_block.ops):
        if op.type == 'fetch':
            need_to_remove_op_index.append(i)
    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)
    program.desc.flush()
    append_fetch_ops(program, fetchs, fetch_holder_name)


def get_ops(pdmodel_prefix):
    # 加载模型
    exe = fluid.Executor(fluid.CPUPlace())
    [prog, feed, fetchs] = paddle.static.load_inference_model(
                        pdmodel_prefix, 
                        exe)

    operator_set = set(())
    # 输出计算图所有结点信息
    for i, op in enumerate(prog.blocks[0].ops):
        #print(i, op.type)
        if op.type not in { 'fetch', 'feed' }:
            operator_set.add(op.type)

    return sorted(operator_set)


if __name__ == '__main__':
    operator_set = get_ops('/home/cecilia/explore/PDPD/PaddleOMZAnalyzer/exporter/paddleclas/MobileNetV3_large_x1_0/inference')

    print(operator_set, len(operator_set))