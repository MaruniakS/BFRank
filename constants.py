# Features to use for classification
FEATURES = [
    'max_as_path_length',
    'av_as_path_length',
    'av_number_of_bits_in_prefix_ipv4',
    'max_number_of_bits_in_prefix_ipv4',
    'max_unique_as_path_length',
    'av_unique_as_path_length',
    'var_as_degree_in_paths',
    'av_as_degree_in_paths',
    'av_number_of_edges_not_in_as_graph',
    'av_number_of_P2P_edges',
    'av_number_of_C2P_edges',
    'av_number_of_P2C_edges',
    'av_number_of_S2S_edges',
    'av_number_of_non_vf_paths',
    'avg_geo_dist_same_bitlen',
    'avg_geo_dist_diff_bitlen',
    'n_announcements'
]

# Class labels mapping
CLASS_LABELS = {0: "Direct", 1: "Indirect", 2: "Outage"}

N_CLASSES = 3
BATCH_SIZE = 1
EPOCHS = 10
SEQUENCE_LENGTH = 10
TRAIN_SIZE = 0.7
TEST_SIZE= 0.1

SEED_SPLIT1 = 0
SEED_SPLIT2 = 1
