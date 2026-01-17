import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def process_images_pytorch(train_data: str, val_data: str = None, split_ratio=0.2):
    '''
    Собирает картинки находящиеся по пути train_data и val_data (если существует)
    Если val_data не существует, то разбивает train_data на train и val
    ПРИМЕР
    '''
    # бла бла делаем dataloader-ы train_loader и val_loader
    train_loader, val_loader = 0, 0
    return train_loader, val_loader


def pytorch_mnist_processor(batch_size: int = 64, val_split: float = 0.2, seed: int = 42):
    """
    Загружает MNIST и делит тренировочную выборку на train и val.
    
    Args:
        batch_size: Размер батча.
        val_split: Доля данных для валидации (0.2 = 20%).
        seed: Число для воспроизводимости разделения.
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # 1. Трансформации (стандартные для MNIST)
    # MNIST — это ч/б картинки 28x28. Мы переводим их в тензоры 
    # и нормализуем (среднее 0.1307, ст.отклонение 0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. Загружаем полный тренировочный сет (60 000 картинок)
    full_train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    # 3. Рассчитываем размеры выборок
    n_total = len(full_train_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    # 4. Разделяем (фиксируем seed через генератор)
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train_dataset, [n_train, n_val], generator=generator)

    # 5. Оборачиваем в DataLoader-ы
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader