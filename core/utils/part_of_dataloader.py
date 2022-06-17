from torch.utils.data import DataLoader, Subset


def part_of_dataloader(data_loader, num_iters):
    dataset = data_loader.dataset
    batchsize = data_loader.batch_size 
    pin_memory = data_loader.pin_memory
    num_workers = data_loader.num_workers
    sub_dataset = Subset(dataset, list(range(0, batchsize*num_iters)))
    return DataLoader(
        sub_dataset, 
        batch_size=batchsize,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )