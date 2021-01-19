import argparse

def p_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default='./data/PEMS04/pems04.npz', type=str)
    parser.add_argument(
        "--data_path_adjacency", default='./data/PEMS04/adjacency.npy', type=str)
    # parser.add_argument(
    #     "--data_path_adjacency", default='./data/adjacency.npy', type=str)
    # parser.add_argument(
    #     "--data_path_adjacency_connected", default='./data/way_adjacency.npy', type=str)
    # parser.add_argument(
    #     "--graph_embedding", default='./data/graph_embedding.npy', type=str)
    # parser.add_argument(
    #     "--od_feature", default='./data/didi_graph_od_feature.npy', type=str)
    # parser.add_argument(
    #     "--data_path_adjacency_poi", default='./data/poi_adjacency.npy', type=str)
    
    parser.add_argument("--seq_len", default=5, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--max_epoches", default=200, type=int)
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--lr", default=0.001, type=float)

    # parser.add_argument("--matrix_size", default=11, type=int)

    args = parser.parse_args()

    return args