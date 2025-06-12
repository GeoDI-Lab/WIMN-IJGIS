import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import argparse
import math
import json
import time
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error


from models.model import WIMN
from Optim import Optim

def de_normalized(interaction_matrix, batch_1):

    return interaction_matrix * (batch_1.max_value.cuda() - batch_1.min_value.cuda()) + batch_1.min_value.cuda()


criterion = nn.CrossEntropyLoss(reduction="sum")

def loss_function(truth, predict):
    RECON = criterion(predict,truth)
    return RECON

def make_dir(path, dic_name):
    path = os.path.join(path, dic_name)
    is_dir_exist = os.path.exists(path)
    if is_dir_exist:
        print("----Dic existed----")
    else:
        os.mkdir(path)
        print("----Dic created successfully----")
    return path

def root_mean_squared_error(generated_flow, real_flow):
    """
    The implementation of RMSE for Deep_gravity
    """
    try:
        rmse = mean_squared_error(generated_flow, real_flow, squared=False)
    except:
        rmse=0
    return rmse

def jesen_shannon_distance(generated_flow, real_flow):
    """
    The implementation of jesen_shannon_distance
    """

    # 确保矩阵是概率分布（每行和为1）
    generated_flow = generated_flow / generated_flow.sum()
    real_flow = real_flow / real_flow.sum()

    # 计算Jensen-Shannon Divergence
    try:
        jsd = jensenshannon(generated_flow, real_flow)
    except:
        jsd=0
    return jsd

def normalized_root_mean_squared_error(predicted_flow, real_flow):
    """
    The implementation of RMSE for Deep_gravity
    """
    try:
        rmse = mean_squared_error(predicted_flow, real_flow, squared=False)

        std_dev = np.std(real_flow)

        nrmse = rmse / std_dev   
    except:
        nrmse = 0
    return nrmse

def common_part_of_commuters(predicted_flow, real_flow, numerator_only=False):
    """
    The implementation of common_part_of_commuters adapted for flattened 2D matrices.
    :param predicted_flow: A 1D numpy array representing the generated flow.
    :param real_flow: A 1D numpy array representing the real flow.
    :param numerator_only: Boolean flag to consider only the numerator in the calculation.
    :return: The value of CPC
    """
    if predicted_flow.shape != real_flow.shape:
        raise ValueError("The shapes of predicted_flow and real_flow must be the same")
    try:
        if numerator_only:
            tot = 1.0
        else:
            tot = np.sum(predicted_flow) + np.sum(real_flow)

        if tot > 0:
            cpc = 2.0 * np.sum(np.minimum(predicted_flow, real_flow)) / tot
        else:
            return 0.0
    except:
        cpc=0
    return cpc

def spearmanr_correlation(predicted_flow, real_flow):
    """_summary_

    Args:
        predicted_flow (_type_): _description_
        real_flow (_type_): _description_
    """
    try:
        spearman_corr, p_value = spearmanr(predicted_flow, real_flow)
    except:
        spearman_corr = 0
    return spearman_corr


def pearsonr_correlation(predicted_flow, real_flow):
    """
    The implementation of pearsonr
    :param predicted_flow:
    :param real_flow:
    :return: The value of corr
    """
    try:
        pearsonr_corr, p_value = pearsonr(predicted_flow, real_flow)
    except:
        pearsonr_corr = 0
    return pearsonr_corr

def symmetric_mean_absolute_percentage_error(predicted_flow, real_flow):
    try:
        smape = 100 * np.mean(2 * np.abs(real_flow - predicted_flow) / (np.abs(real_flow) + np.abs(predicted_flow)))
    except:
        smape=0
    return smape

def compute_metric(p, r):

    cpc = common_part_of_commuters(p, r)
    p_corr = pearsonr_correlation(p, r)
    s_corr = spearmanr_correlation(p, r)

    jsd = jesen_shannon_distance(p, r)
    rmse = root_mean_squared_error(p, r)
    nrmse = normalized_root_mean_squared_error(p, r)
    smape = symmetric_mean_absolute_percentage_error(p, r)

    return cpc, p_corr, s_corr, jsd, rmse, nrmse, smape


def calculate_segment_metrics(predicted_values, real_values, num_segments=5):
    """
    Divide the array into a specified number of segments from large to small, and then calculate the RMSE and correlation coefficient of each segment.

    Parameters:
    - real_values: array of real values.
    - predicted_values: array of predicted values.
    - num_segments: number of segments to be divided, default is 5.

    Returns:
    - segment_metrics: list of RMSE and correlation coefficients for each segment, in the format of [sub_metrics1, sub_metrics2, ...].
    """
    num_elements_per_segment = len(real_values) // num_segments
    sorted_indices = np.argsort(real_values)[::-1]

    segments = []
    for i in range(3):
        start_index = i * num_elements_per_segment
        end_index = (i + 1) * num_elements_per_segment
        segment_indices = sorted_indices[start_index:end_index]

        real_segment = real_values[segment_indices]
        predicted_segment = predicted_values[segment_indices]

        segments.append([predicted_segment, real_segment])

    return segments


def evaluate(dataloader_node, dataloader_snow, model, mode):
    """
    Evaluating or testing function
    :param data:
    :param X:
    :param Y:
    :param model:
    :param batch_size:
    :return:
    """
    model.eval()
    
    predicted_flow = []
    real_flow = []

    for batch_1, batch_2  in zip(dataloader_node, dataloader_snow):

        node_features = Variable(batch_1.x).cuda().float()
        snow_features = []
        snow_features.append(Variable(batch_2.x).cuda().float())
        snow_features.extend([Variable(i).cuda().float() for i in batch_2.hub_top_x])


        
        edge_index = []
        edge_index.append(Variable(batch_1.edge_index).cuda().long())
        edge_index.extend(Variable(i).cuda().long() for i in batch_1.hub_top_edge_index)

        interaction_index = [Variable(i).cuda() for i in batch_1.interaction_index]

        adjacency_index = [Variable(i).cuda().long() for i in batch_1.adjacency_edge_index]
        edge_attr = Variable(batch_1.edge_attr).cuda().float()
    

        hub_indices_list = [Variable(i).cuda().long() for i in batch_1.hub_indices_lists]
        interaction_indices_lists = [Variable(i).cuda().int() for i in batch_1.interaction_indices_lists]
        adjacency_indices_lists = [Variable(i).cuda().int() for i in batch_1.adjacency_indices_lists]
        hub_top_edge_mask = [Variable(i).cuda() for i in batch_1.hub_top_edge_mask]
        # print(edge_unique_indices)
        # print(type(edge_unique_indices))

        predict_interaction = model(edge_index = edge_index,
                                    interaction_index = interaction_index,
                                    edge_attr_unique = edge_attr, 
                                    node_input_x = node_features, 
                                    weather_inputs_x = snow_features,
                                    adjacency_index = adjacency_index, 
                                    hub_indices_list = hub_indices_list, 
                                    interaction_indices_lists = interaction_indices_lists,
                                    adjacency_indices_lists = adjacency_indices_lists,
                                    hub_top_edge_mask = hub_top_edge_mask) #(9)
        
        next_interaction_prob = Variable(batch_1.y).cuda().float()
        y_sum = Variable(batch_1.y_sum).cuda().float()
        predict_interaction_prob = nn.functional.softmax(predict_interaction, dim = -1)
        predict_interaction = predict_interaction_prob * y_sum
        next_interaction = next_interaction_prob * y_sum

        next_interaction = de_normalized(next_interaction, batch_1)
        predict_interaction = de_normalized(predict_interaction, batch_1)

        if mode == "test":
            npy_path = make_dir(make_dir(make_dir(path_result, "results_csv"), batch_1.next_date_time[0]), batch_1.graph_index[0])
            np.savetxt(os.path.join(npy_path, "predict.csv"), predict_interaction.cpu().data.numpy())
            np.savetxt(os.path.join(npy_path, "true.csv"), next_interaction.cpu().data.numpy())
        predicted_flow.append(predict_interaction.cpu().detach().numpy())
        real_flow.append(next_interaction.cpu().numpy())

    
   # Initialize metrics lists
    metrics_lists = [0, 0, 0, 0, 0, 0, 0]
    segment_metrics = [[0, 0, 0, 0, 0, 0, 0] for _ in range(3)] # For 5 segments

    metrics_str_lists = ["CPC", "P_CORR", "S_CORR", "JSD", "RMSE", "NRMSE", "SMAPE"]
    for p, r in zip(predicted_flow, real_flow):
    
        sub_metrics = compute_metric(p, r)
        metrics_lists = [ i+j for i, j in zip(metrics_lists, sub_metrics)]
        
        segments = calculate_segment_metrics(p, r)  # Calculate metrics for segments
        for index, segment in enumerate(segments):
            sub_metrics = compute_metric(segment[0], segment[1])
            segment_metrics[index] = [i + j for i, j in zip(segment_metrics[index], sub_metrics)] # Accumulate segment metrics

        
    metrics_lists = [x / len(predicted_flow) for x in metrics_lists]
    segment_metrics = [[x / len(predicted_flow) for x in segment] for segment in segment_metrics]

    if mode == "test":

        with open(os.path.join(path_result, "results-WIMN.txt"), "a") as result_file:
            result_file.write("The whole testing set\r\n")
            for string, value in zip(metrics_str_lists, metrics_lists):
                result_file.write("{}:{}\r\n".format(string, value))
            for i, segment in enumerate(segment_metrics, start=1):
                result_file.write(f"The {i}th segment set\r\n")
                for string, value in zip(metrics_str_lists, segment):
                    result_file.write("{}:{}\r\n".format(string, value))
            
            
    return tuple(metrics_lists)



def diag_flatten_function(matrix):

    mask = torch.triu(torch.ones_like(matrix), diagonal=1).bool()
    upper_triangle_no_diag_flattened = torch.masked_select(matrix, mask)
    return upper_triangle_no_diag_flattened


def train(dataloader_node, dataloader_snow, model):
    """
    Training function
    :param data:
    :param 
    :param model:
    :param optim:
    :param batch_size:
    :return:
    """
    model.train()


    predicted_flow = []
    real_flow = []
    
    i = 1
    batch_loss = 0.0
    total_loss = 0.0
    
    for batch_1, batch_2  in zip(dataloader_node, dataloader_snow):
        if i % args.batch_size == 0 or i == len(train_location):
            model.zero_grad()
        node_features = Variable(batch_1.x).cuda().float()
        snow_features = []
        snow_features.append(Variable(batch_2.x).cuda().float())
        snow_features.extend([Variable(i).cuda().float() for i in batch_2.hub_top_x])


        
        edge_index = []
        edge_index.append(Variable(batch_1.edge_index).cuda().long())
        edge_index.extend(Variable(i).cuda().long() for i in batch_1.hub_top_edge_index)

        interaction_index = [Variable(i).cuda() for i in batch_1.interaction_index]

        adjacency_index = [Variable(i).cuda().long() for i in batch_1.adjacency_edge_index]
        edge_attr = Variable(batch_1.edge_attr).cuda().float()
    

        hub_indices_list = [Variable(i).cuda().long() for i in batch_1.hub_indices_lists]
        interaction_indices_lists = [Variable(i).cuda().int() for i in batch_1.interaction_indices_lists]
        adjacency_indices_lists = [Variable(i).cuda().int() for i in batch_1.adjacency_indices_lists]
        hub_top_edge_mask = [Variable(i).cuda() for i in batch_1.hub_top_edge_mask]
        # print(edge_unique_indices)
        # print(type(edge_unique_indices))

        predict_interaction = model(edge_index = edge_index,
                                    interaction_index = interaction_index,
                                    edge_attr_unique = edge_attr, 
                                    node_input_x = node_features, 
                                    weather_inputs_x = snow_features,
                                    adjacency_index = adjacency_index, 
                                    hub_indices_list = hub_indices_list, 
                                    interaction_indices_lists = interaction_indices_lists,
                                    adjacency_indices_lists = adjacency_indices_lists,
                                    hub_top_edge_mask = hub_top_edge_mask) #(9)
        
        next_interaction_prob = Variable(batch_1.y).cuda().float()

        interation_loss = loss_function(truth = next_interaction_prob, 
                                        predict = predict_interaction)
        
        batch_loss += interation_loss

        y_sum = Variable(batch_1.y_sum).cuda().float()
        predict_interaction_prob = nn.functional.softmax(predict_interaction, dim = -1)
        predict_interaction = predict_interaction_prob * y_sum
        next_interaction = next_interaction_prob * y_sum


        if i % args.batch_size == 0 or i == len(train_location):
            batch_loss.backward() 
            optimizer.step()
            total_loss += batch_loss.cpu().data.numpy()
            batch_loss = 0.0

        i+=1
        next_interaction = de_normalized(next_interaction, batch_1)
        predict_interaction = de_normalized(predict_interaction, batch_1)

        real_flow.append(next_interaction.cpu().numpy())
        predicted_flow.append(predict_interaction.cpu().detach().numpy())

    metrics_lists = [0, 0, 0, 0, 0, 0, 0]                             
    for p, r in zip(predicted_flow, real_flow):
    
        sub_metrics = compute_metric(p, r)
        metrics_lists = [ i+j for i, j in zip(metrics_lists, sub_metrics)]

    metrics_lists = [x / len(predicted_flow) for x in metrics_lists]

    return total_loss, tuple(metrics_lists)


parser = argparse.ArgumentParser(description='WIMN on TCMA')
### hyper-parameters
parser.add_argument('--train_p', type=float, default=0.6, help="the proportion of training dataset")
parser.add_argument('--valid_p', type=float, default=0.2, help="the proportion of validing dataset")
parser.add_argument('--epochs', type=int, default=5000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--seed', type=int, default=54321, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help="The Index of GPU where we want to run the code")
parser.add_argument("--order", type = str, default= "4th", help = "The order of neighborhood")
parser.add_argument('--cuda', type=str, default=True)

parser.add_argument("--node_hidden_dims", type=int, default = [32, 64, 64], help = "the node hidden dimentions")
parser.add_argument("--edge_hidden_dims", type=int, default = [32, 64, 64], help = "the edge hidden dimentions")
parser.add_argument("--pareto_k", type=float, default = 0.2, help = "the proportion of hub nodes")
parser.add_argument("--dropout_rate", type=float, default = 0.0, help = "The dropout rate")

parser.add_argument('--normalize', type=int, default=2, help="The way of normalization during the data preprocessing")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip', type=float, default=10., help='gradient_visual clipping')
parser.add_argument('--optim', type=str, default='adam')

args = parser.parse_args()


# location, path, dictionary 
print("----Creating testing result folder----")
path = "results//"
path = make_dir(path, "{}_neighborhood".format(args.order))
path_seed = make_dir(path, str(args.seed))
path_result = make_dir(path_seed, str(args.dropout_rate))
path_log = make_dir(path_result, "log")
path_model = make_dir(path_result, "pkl")
path_result = make_dir(path_result, "results")

print("----Loading data----")

train_location = torch.load('data//train_location.pt')
train_weather = torch.load('data//train_weather.pt')
valid_location = torch.load('data//valid_location.pt')
valid_weather = torch.load('data/valid_weather.pt')
test_location = torch.load('data//test_location.pt')
test_weather = torch.load('data//test_weather.pt')


node_loader_train = DataLoader(dataset = train_location, 
                               batch_size = 1)

snow_loader_train = DataLoader(dataset = train_weather, 
                               batch_size = 1)

node_loader_valid = DataLoader(dataset = valid_location, 
                               batch_size = 1)
snow_loader_valid = DataLoader(dataset = valid_weather, 
                               batch_size=1)

node_loader_test = DataLoader(dataset = test_location, 
                              batch_size = 1)
snow_loader_test = DataLoader(dataset = test_weather, 
                              batch_size = 1)



# The setting of GPU
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

print("----Building models----")
model = WIMN(node_input_dim = 12, 
             weather_input_dim = 4 + 4, 
             node_hidden_dims = args.node_hidden_dims,
             edge_input_dim = 3,
             edge_hidden_dims = args.edge_hidden_dims,
             dropout_rate = args.dropout_rate)

if args.cuda:
    model.cuda()

nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('----number of parameters: %d----' % nParams)

# saving hyperparamter configurations
args_dict = vars(args)
args_dict["number of parameters"] = nParams
with open(os.path.join(path_log, 'args_config.json'), 'w') as f:
    json.dump(args_dict, f)


optimizer = Optim(params = model.parameters(), method = args.optim, lr = args.lr, max_grad_norm = args.clip)

best_valid_cpc = 0.0
writer = SummaryWriter(path_log)
try:
    print('----Traning begin----')
    last_update = 1 
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        total_loss, train_metrics = train(node_loader_train, snow_loader_train, model)
        valid_metrics = evaluate(node_loader_valid, snow_loader_valid, model, "valid")

        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.4f} | train cpc {:5.4f} | train p_corr {:5.4f} | train s_corr {:5.4f} | train jsd {:5.4f} | train rmse {:5.4f} | train nrmse {:5.4f} | train smape {:5.4f} | valid cpc {:5.4f} | valid p_corr {:5.4f} | valid s_corr {:5.4f} | valid jsd {:5.4f} | valid rmse {:5.4f} | valid nrmse {:5.4f} | valid smape {:5.4f}'.format(
            epoch, (time.time() - epoch_start_time), total_loss, *train_metrics, *valid_metrics))

        metric_prefixes = {
            "train": train_metrics,
            "valid": valid_metrics}

        for prefix, metrics in metric_prefixes.items():
            for metric, value in zip(['cpc', 'p_corr', 's_corr', 'jsd', 'rmse', 'nrmse', 'smape'], metrics):
                writer.add_scalar(f"{prefix}_{metric}", value, epoch)

        with open(os.path.join(path_log, "log.txt"), "a") as file:
                file.write('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.4f} | train cpc {:5.4f} | train p_corr {:5.4f} | train s_corr {:5.4f} | train jsd {:5.4f} | train rmse {:5.4f} | train nrmse {:5.4f} | train smape {:5.4f} | valid cpc {:5.4f} | valid p_corr {:5.4f} | valid s_corr {:5.4f} | valid jsd {:5.4f} | valid rmse {:5.4f} | valid nrmse {:5.4f} | valid smape {:5.4f}\r\n'.format(
            epoch, (time.time() - epoch_start_time), total_loss, *train_metrics, *valid_metrics))
        valid_cpc = valid_metrics[0]
        if best_valid_cpc < valid_cpc:
            
            print("----epoch:{}, save the model----".format(epoch))
            with open(os.path.join(path_model, "model.pkl"), 'wb') as f:
                torch.save(model, f)
            with open(os.path.join(path_log, "log.txt"), "a") as file:
                file.write("----epoch:{}, save the model----\r\n".format(epoch))
            best_valid_cpc = valid_cpc
            last_update = epoch

        if epoch - last_update == 200:
            break
except KeyboardInterrupt:
    print('-' * 90)
    print('----Exiting from training early----')

print("----Testing begin----")
with open(os.path.join(path_model, "model.pkl"), 'rb') as f:
    model = torch.load(f)
    cpc, p_corr, s_corr, jsd, rmse, nrmse, smape = evaluate(node_loader_test, snow_loader_test, model, "test")
print("test cpc {:5.4f} | test p_corr {:5.4f} | test s_corr {:5.4f} | test jsd {:5.4f} | test rmse {:5.4f} | test nrmse {:5.4f} | test smape {:5.4f}".format(cpc, p_corr, s_corr, jsd, rmse, nrmse, smape))


