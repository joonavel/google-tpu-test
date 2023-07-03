import argparse
import tensorflow as tf


if __name__ == 'main':
    # 인자값을 받을 수 있는 인스턴스 생성
    argparser = argparse.ArgumentParser(description='Test')
    argparser.add_argument('--arg1', type=str, required=True)
    argparser.add_argument("--arg2", type=int, required=True)
    argparser.add_argument('--arg3', type=str, default='arg3')
    
    test_args = argparser.parse_args()
    print(f'arg1: {test_args.arg1}\n arg2: {test_args.arg2}\n arg3: {test_args.arg3}')
    
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.TPUStrategy(tpu)

    print(f"Available number of replicas: {strategy.num_replicas_in_sync}")