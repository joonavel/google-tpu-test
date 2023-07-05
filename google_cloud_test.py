import argparse
import tensorflow as tf


# check TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

strategy = tf.distribute.TPUStrategy(tpu)

print(f"Available number of replicas: {strategy.num_replicas_in_sync}")

def _parse_function(example_proto):

    name_to_features = {'id': tf.io.FixedLenFeature([], tf.string),
                        'document': tf.io.FixedLenFeature([], tf.string),
                        'summary': tf.io.FixedLenFeature([], tf.string)}

    example = tf.io.parse_single_example(example_proto, name_to_features)

    for name in list(example.keys()):
        t = example[name]
        t = tf.cast(t, tf.string)
        example[name] = t

    return example


def load_tfrecord_dataset(tfrecord_name, batch_size, shuffle=True, buffer_size=10240):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()

    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)


    dataset = raw_dataset.map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )



    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset



if __name__ == 'main':
    # Check args
    argparser = argparse.ArgumentParser(description='Test')
    argparser.add_argument('--arg1', type=str, required=True)
    argparser.add_argument("--arg2", type=int, required=True)
    argparser.add_argument('--arg3', type=str, default='arg3')
    
    test_args = argparser.parse_args()
    print(f'arg1: {test_args.arg1}\n arg2: {test_args.arg2}\n arg3: {test_args.arg3}')
    
    # check TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.TPUStrategy(tpu)

    print(f"Available number of replicas: {strategy.num_replicas_in_sync}")
    
    # Check Access to Google Cloud Storage
    tfr_path = 'gs://ohsori-tfrecord/tfrecord/literature.tfrecords'
    tf_record = load_tfrecord_dataset(tfr_path, batch_size=strategy.num_replicas_in_sync, shuffle=False)
    tf_record = iter(tf_record)
    
    print([summary.decode('utf-8') for summary in next(tf_record)['summary'].numpy()])