import random
from PIL import Image
from torch.utils.data import Dataset

from utils.jigsaw_utils import create_jigsaw_image, create_grid_jigsaw_input, create_box_jigsaw_input


class JigsawDataset(Dataset):
    def __init__(
        self,
        image_paths,
        stage,
        m,
        n,
        n_c,
        shuffle_mn,
        mask_ratio,
        width_min,
        width_max,
        height_min,
        height_max,
        forward,
        question_type,
        think,
        instruct_model
    ):
        super(JigsawDataset, self).__init__()

        self.image_paths = image_paths
        self.stage = [0] + stage
        self.m = m
        self.n = n
        self.n_c = n_c
        self.shuffle_mn = shuffle_mn
        self.mask_ratio = mask_ratio
        self.width_min = width_min
        self.width_max = width_max
        self.height_min = height_min
        self.height_max = height_max
        self.forward = forward
        self.question_type = question_type
        self.think = think
        self.instruct_model = instruct_model
        self.count = 0

        assert len(self.m) == len(self.n) == len(self.n_c) == len(self.stage) - 1


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, i):
        random.seed(i)

        image_path = self.image_paths[i]
        image = Image.open(image_path).convert("RGB")

        self.count += 1
        stage = 0
        for s in range(len(self.stage)):
            if self.count > self.stage[s]:
                stage = s

        m = self.m[stage]
        n = self.n[stage]

        if self.shuffle_mn and random.random() < 0.5:
            m, n = n, m

        choices = [i for i in range(1, m * n + 1)]
        k = random.choice(choices)

        result = create_jigsaw_image(
            image,
            m,
            n,
            k,
            self.mask_ratio,
            self.width_min,
            self.width_max,
            self.height_min,
            self.height_max,
            self.question_type
        )

        if self.question_type != "box":
            image_jigsaw, patches_forward, patches_backward = result
            input = create_grid_jigsaw_input(
                image_jigsaw,
                patches_forward,
                patches_backward,
                m,
                n,
                self.n_c,
                self.mask_ratio,
                self.forward,
                self.question_type,
                self.think,
                self.instruct_model
            )
        else:
            jigsaw_image, box_gt = result
            input = create_box_jigsaw_input(
                jigsaw_image,
                box_gt,
                m,
                n,
                k,
                self.mask_ratio,
                self.forward,
                self.think,
                self.instruct_model
            )

        input["image_id"] = image_path.split("/")[-1]

        return input
