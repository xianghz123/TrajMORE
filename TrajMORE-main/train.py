import argparse
from grid2vec import *
from TrajMORE import *
import utils
import parameters

timer = utils.Timer()

torch.cuda.empty_cache()


def log_to_file(message, log_file='data/train_log/train_result'):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def train_TrajMORE(args):
    log_file = 'data/train_log/train_result'

    # load args
    train_dataset = args.train_dataset
    validate_dataset = args.validate_dataset
    test_dataset = args.test_dataset
    batch_size = args.batch_size
    pretrained_embedding_file = args.pretrained_embedding
    emb_size = args.embedding_size
    learning_rate = args.learning_rate
    epochs = args.epoch_num
    cp = args.checkpoint
    vocab_size = args.vocab_size
    loss_func_name = args.loss_func
    sampler_name = args.sampler
    triplet_num = args.triplet_num
    neg_rate = args.neg_rate
    dataset_size = args.dataset_size
    encoder_layers = args.encoder_layers
    min_len = args.min_len
    cumulative_iters = args.cumulative_iters
    max_len = args.max_len
    heads = args.heads
    attention_gru_hidden_dim = args.attention_gru_hidden_dim
    device = torch.device(args.device)
    vp = args.visdom_port

    # prepare data
    timer.tik("prepare data")
    log_to_file("prepare data start", log_file)
    dataset = MetricLearningDataset(train_dataset, triplet_num, min_len, max_len,
                                    dataset_size=dataset_size, neg_rate=neg_rate)
    if sampler_name == "seq_hard":
        sampler = SeqHardSampler(dataset, batch_size)
        train_loader = tud.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)
    else:
        train_loader = tud.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    validate_dataset = MetricLearningDataset(validate_dataset, triplet_num, min_len=0,
                                             max_len=99999, dataset_size=None, neg_rate=neg_rate)
    validate_loader = tud.DataLoader(validate_dataset, batch_size=len(validate_dataset), collate_fn=collate_fn)
    validate_train_loader = tud.DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)

    # Load new test dataset
    test_dataset = MetricLearningDataset(test_dataset, triplet_num, min_len=0,
                                         max_len=99999, dataset_size=None, neg_rate=neg_rate)
    test_loader = tud.DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=collate_fn)

    log_to_file(f"prepare data done, {timer.tok('prepare data')} after prepare data start", log_file)

    # init t2g
    from traj2grid import Traj2Grid
    from parameters import min_lon, max_lon, min_lat, max_lat
    str_grid2idx = json.load(open(args.grid2idx))
    grid2idx = {eval(g): str_grid2idx[g] for g in list(str_grid2idx)}
    t2g = Traj2Grid(400, 400, min_lon, min_lat, max_lon, max_lat, grid2idx)

    # init model
    timer.tik("init model")
    log_to_file("init model start", log_file)
    pre_emb = None
    if pretrained_embedding_file:
        pre_emb = torch.FloatTensor(np.load(pretrained_embedding_file))
    model = TrajMORE(vocab_size, emb_size, heads, pre_emb=pre_emb, attention_gru_hidden_dim=attention_gru_hidden_dim,
                    encoder_layers=encoder_layers, t2g=t2g).to(device)
    model.mean_x = dataset.meanx
    model.mean_y = dataset.meany
    model.std_x = dataset.stdx
    model.std_y = dataset.stdy
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    epoch_start = 0
    log_to_file(f"init model done, {timer.tok('init model')} after init model start", log_file)

    # loss_func
    if loss_func_name == "seq_hard":
        loss_func = model.calculate_loss_seq_sampler
    else:
        loss_func = model.calculate_loss_vanilla

    # load checkpoint
    if cp is not None:
        cp = torch.load(cp)
        if cp.get('model'):
            model.load_state_dict(cp['model'])
        if cp.get('optimizer'):
            optimizer.load_state_dict(cp['optimizer'])
        if cp.get('epoch'):
            epoch_start = cp['epoch'] + 1

    # init visdom
    if vp != 0:
        from visdom import Visdom
        env = Visdom(port=args.visdom_port)
        pane1_name = f'train_loss_{timer.now()}'
        pane2_name = f'validate_acc_{timer.now()}'
        pane3_name = f'scatter200_{timer.now()}'

    # train
    batch_count = 0
    best_rank = 99999
    best_hr10 = 0
    best_hr20 = 0
    best_r10_50 = 0
    best_r10_100 = 0
    for epoch in range(epoch_start, epochs):
        for batch_idx, (anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                        trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                        anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a) in enumerate(train_loader):
            anchor = anchor.to(device)
            pos = pos.to(device)
            trajs_a = trajs_a.to(device)
            trajs_p = trajs_p.to(device)
            sim_pos = sim_pos.to(device)
            sim_matrix_a = sim_matrix_a.to(device)
            loss_map = dataset.loss_map
            loss, loss_p, loss_n = loss_func(
                anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a, loss_map)
            loss = loss / cumulative_iters
            loss.backward()
            if batch_idx % cumulative_iters == 0:
                optimizer.step()
                optimizer.zero_grad()
            log_message = f"epoch:{epoch} batch:{batch_idx} train loss:{loss.item()} lamb:{model.lamb.item()}"
            log_to_file(log_message, log_file)
            print(log_message)
            batch_count += 1
            if vp != 0:
                env.line(X=[batch_count], Y=[loss.item()], win=pane1_name, name="train loss", update='append')
                env.line(X=[batch_count], Y=[loss_p.item()], win=pane1_name, name="train loss_p", update='append')
                env.line(X=[batch_count], Y=[loss_n.item()], win=pane1_name, name="train loss_n", update='append')

        # Evaluate on validation set
        rank_validate, hr_10_validate, hr_20_validate, r10_50_validate, r10_100_validate, pca_x_validate = model.evaluate(
            validate_loader, device, tri_num=triplet_num)
        rank_train, hr_10_train, hr_20_train, r10_50_train, r10_100_train, pca_x_train = model.evaluate(
            validate_train_loader, device, tri_num=triplet_num)

        # Evaluate on test set
        rank_test, hr_10_test, hr_20_test, r10_50_test, r10_100_test, pca_x_test = model.evaluate(test_loader, device,
                                                                                                  tri_num=triplet_num)

        if vp != 0:
            env.line(X=[epoch], Y=[rank_validate], win=pane2_name, name="Rank", update='append')
            env.line(X=[epoch], Y=[hr_10_validate], win=pane2_name, name="HR10", update='append')
            env.line(X=[epoch], Y=[hr_20_validate], win=pane2_name, name="HR20", update='append')
            env.line(X=[epoch], Y=[r10_50_validate], win=pane2_name, name="R10@50", update='append')
            env.line(X=[epoch], Y=[r10_100_validate], win=pane2_name, name="R10@100", update='append')

            env.line(X=[epoch], Y=[rank_test], win=pane2_name, name="Test Rank", update='append')
            env.line(X=[epoch], Y=[hr_10_test], win=pane2_name, name="Test HR10", update='append')
            env.line(X=[epoch], Y=[hr_20_test], win=pane2_name, name="Test HR20", update='append')
            env.line(X=[epoch], Y=[r10_50_test], win=pane2_name, name="Test R10@50", update='append')
            env.line(X=[epoch], Y=[r10_100_test], win=pane2_name, name="Test R10@100", update='append')

            env.line(X=[epoch], Y=[hr_10_train], win=pane2_name, name="HR10", update='append')
            env.line(X=[epoch], Y=[hr_20_train], win=pane2_name, name="HR20", update='append')
            env.line(X=[epoch], Y=[r10_50_train], win=pane2_name, name="R10@50", update='append')
            env.line(X=[epoch], Y=[r10_100_train], win=pane2_name, name="R10@100", update='append')
            # env.scatter(X=pca_x, win=pane3_name, name="pca_xy", opts=dict(markersize=5, markersymbol='cross-thin-open'))

        log_message = f"epoch:{epoch} rank_test:{rank_test:.4f}, hr_10_test:{hr_10_test:.4f}, hr_20_test:{hr_20_test:.4f}, r10_50_test:{r10_50_test:.4f} r10_100_test:{r10_100_test:.4f} max_loss_map:{loss_map.max():.6f}"
        log_to_file(log_message, log_file)
        print(log_message)

        log_message = f"epoch:{epoch} rank_train:{rank_train:.4f}, hr_10_train:{hr_10_train:.4f}, hr_20_train:{hr_20_train:.4f}, r10_50_train:{r10_50_train:.4f} r10_100_train:{r10_100_train:.4f} max_loss_map:{loss_map.max():.6f}"
        log_to_file(log_message, log_file)
        print(log_message)

        log_message = f"epoch:{epoch} rank_validate:{rank_validate:.4f}, hr_10_validate:{hr_10_validate:.4f}, hr_20_validate:{hr_20_validate:.4f}, r10_50_validate:{r10_50_validate:.4f} r10_100_validate:{r10_100_validate:.4f} max_loss_map:{loss_map.max():.6f}"
        log_to_file(log_message, log_file)
        print(log_message)

        if epoch % 10 == 9:
            cp = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(cp, f'model/cp_{epoch}_loss{round(float(loss), 3)}_rank_{round(float(rank_validate), 3)}.pth')

        if rank_validate < best_rank:
            best_rank = rank_validate
            best_hr10 = hr_10_validate
            best_hr20 = hr_20_validate
            best_r10_50 = r10_50_validate
            best_r10_100 = r10_100_validate
            model.to("cpu")
            torch.save(model, f'model/best_model.pth')
            model.to(device)
            log_message = f"save new best rank:{best_rank}"
            log_to_file(log_message, log_file)
            print(log_message)

    log_message = f"train finish. best rank:{best_rank} hr10:{best_hr10} best hr20:{best_hr20} best r10@50:{best_r10_50} best r10@100:{best_r10_100}"
    log_to_file(log_message, log_file)
    print(log_message)

parser = argparse.ArgumentParser(description="train.py")
parser.add_argument('--model', '-m', type=str, default='TrajMORE', help='model name')
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--cumulative_iters', '-cumu', type=int, default=1)
parser.add_argument('--epoch_num', '-ep', type=int, default=100)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
parser.add_argument('--visdom_port', '-vp', type=int, default=0)
parser.add_argument('--checkpoint', '-cp', type=str, default=None)
parser.add_argument('--device', '-device', type=str, default="cuda")

parser.add_argument('--grid2idx', '-dict', type=str, default="data/grid/str_grid2idx_400_31925.json")
parser.add_argument('--train_dataset', '-data_tr', type=str,
                    default="data/train/train_frechet_dataset.json")
parser.add_argument('--validate_dataset', '-data_va', type=str,
                    default="data/valid/valid_frechet_dataset.json")
parser.add_argument('--test_dataset', '-data_nv', type=str,
                    default="data/test/test_discret_frechet.json")

parser.add_argument('--pretrained_embedding', '-pre', type=str, default=None)
parser.add_argument('--loss_func', '-loss_func', type=str, default="vanilla")
parser.add_argument('--sampler', '-sampler', type=str, default="vanilla")
parser.add_argument('--embedding_size', '-emb', type=int, default=128)
parser.add_argument('--vocab_size', '-vocab', type=int, default=31925)
parser.add_argument('--dataset_size', '-data_size', type=int, default=None)
parser.add_argument('--min_len', '-min_len', type=int, default=0)
parser.add_argument('--max_len', '-max_len', type=int, default=99999)
parser.add_argument('--window_size', '-ws', type=int, default=20)
parser.add_argument('--triplet_num', '-tn', type=int, default=10)
parser.add_argument('--neg_rate', '-neg_rate', type=int, default=10)
parser.add_argument('--attention_gru_hidden_dim', '-att_gru_hid', type=int, default=128)  
parser.add_argument('--encoder_layers', '-encoder_layers', type=int, default=2)
parser.add_argument('--heads', '-heads', type=int, default=4)

args = parser.parse_args()

args_message = str(args)
print(args_message)
log_to_file(args_message, 'data/train_log/train_result')

if args.model == "TrajMORE":
    train_TrajMORE(args)
elif args.model == "grid2vec":
    train_grid2vec(args.grid2idx, args.window_size, args.embedding_size, args.batch_size, args.epoch_num,
                   args.learning_rate, args.checkpoint, args.visdom_port)
else:
    raise ValueError("model must be grid2vec or TrajMORE")
