from preprocess import load_train_data
from dataset import BPRDataset
from model import BPR
from trainer import BPRTrainer
from torch.utils.data import DataLoader, random_split, default_collate


def main(config):
    inter_data, data_info = load_train_data()

    dataset = BPRDataset(inter_data)
    train_dataset, valid_dataset = random_split(dataset, (0.7, 0.3))
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=default_collate,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=default_collate,
    )

    model = BPR(
        n_users=data_info["users"],
        n_items=data_info["items"],
        embedding_dim=config["embedding_dim"],
    )
    trainer = BPRTrainer(
        model=model,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        device=config["device"],
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    trainer.train()


if __name__ == "__main__":
    import torch

    config = dict()

    config["batch_size"] = 64
    config["embedding_dim"] = 256
    config["epochs"] = 100
    config["learning_rate"] = 0.01
    config["weight_decay"] = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    main(config)
