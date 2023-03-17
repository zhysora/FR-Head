import numpy as np

from torch.utils.data import Dataset

from feeders import tools

ntu120_class_name = [
    "A1. drink water", "A2. eat meal/snack", "A3. brushing teeth", "A4. brushing hair", "A5. drop", "A6. pickup",
    "A7. throw", "A8. sitting down", "A9. standing up (from sitting position)", "A10. clapping", "A11. reading",
    "A12. writing", "A13. tear up paper", "A14. wear jacket", "A15. take off jacket", "A16. wear a shoe",
    "A17. take off a shoe", "A18. wear on glasses", "A19. take off glasses", "A20. put on a hat/cap",
    "A21. take off a hat/cap", "A22. cheer up", "A23. hand waving", "A24. kicking something", "A25. reach into pocket",
    "A26. hopping (one foot jumping)", "A27. jump up", "A28. make a phone call/answer phone", "A29. playing with phone/tablet",
    "A30. typing on a keyboard", "A31. pointing to something with finger", "A32. taking a selfie", "A33. check time (from watch)",
    "A34. rub two hands together", "A35. nod head/bow", "A36. shake head", "A37. wipe face", "A38. salute", "A39. put the palms together",
    "A40. cross hands in front (say stop)", "A41. sneeze/cough", "A42. staggering", "A43. falling", "A44. touch head (headache)",
    "A45. touch chest (stomachache/heart pain)", "A46. touch back (backache)", "A47. touch neck (neckache)", "A48. nausea or vomiting condition",
    "A49. use a fan (with hand or paper)/feeling warm", "A50. punching/slapping other person", "A51. kicking other person",
    "A52. pushing other person", "A53. pat on back of other person", "A54. point finger at the other person",
    "A55. hugging other person", "A56. giving something to other person", "A57. touch other person's pocket",
    "A58. handshaking", "A59. walking towards each other", "A60. walking apart from each other", "A61. put on headphone",
    "A62. take off headphone", "A63. shoot at the basket", "A64. bounce ball", "A65. tennis bat swing", "A66. juggling table tennis balls",
    "A67. hush (quite)", "A68. flick hair", "A69. thumb up", "A70. thumb down", "A71. make ok sign", "A72. make victory sign",
    "A73. staple book", "A74. counting money", "A75. cutting nails", "A76. cutting paper (using scissors)", "A77. snapping fingers",
    "A78. open bottle", "A79. sniff (smell)", "A80. squat down", "A81. toss a coin", "A82. fold paper", "A83. ball up paper",
    "A84. play magic cube", "A85. apply cream on face", "A86. apply cream on hand back", "A87. put on bag", "A88. take off bag",
    "A89. put something into a bag", "A90. take something out of a bag", "A91. open a box", "A92. move heavy objects", "A93. shake fist",
    "A94. throw up cap/hat", "A95. hands up (both hands)", "A96. cross arms", "A97. arm circles", "A98. arm swings", "A99. running on the spot",
    "A100. butt kicks (kick backward)", "A101. cross toe touch", "A102. side kick", "A103. yawn", "A104. stretch oneself",
    "A105. blow nose", "A106. hit other person with something", "A107. wield knife towards other person",
    "A108. knock over other person (hit with body)", "A109. grab other person’s stuff", "A110. shoot at other person with a gun",
    "A111. step on foot", "A112. high-five", "A113. cheers and drink", "A114. carry something with other person",
    "A115. take a photo of other person", "A116. follow other person", "A117. whisper in other person’s ear",
    "A118. exchange things with other person", "A119. support somebody with hand", "A120. finger-guessing game (playing rock-paper-scissors)"
]

ntu120_class_name_short = [
    "A1. drink water", "A2. eat meal", "A3. brushing teeth", "A4. brushing hair", "A5. drop", "A6. pickup",
    "A7. throw", "A8. sitting down", "A9. standing up (from sitting position)", "A10. clapping", "A11. reading",
    "A12. writing", "A13. tear up paper", "A14. wear jacket", "A15. take off jacket", "A16. wear a shoe",
    "A17. take off a shoe", "A18. wear on glasses", "A19. take off glasses", "A20. put on a hat",
    "A21. take off a hat", "A22. cheer up", "A23. hand waving", "A24. kicking something", "A25. reach into pocket",
    "A26. hopping (one foot jumping)", "A27. jump up", "A28. make a phone call", "A29. playing with phone",
    "A30. typing on a keyboard", "A31. pointing to something with finger", "A32. taking a selfie", "A33. check time (from watch)",
    "A34. rub two hands together", "A35. nod head", "A36. shake head", "A37. wipe face", "A38. salute", "A39. put the palms together",
    "A40. cross hands in front (say stop)", "A41. sneeze", "A42. staggering", "A43. falling", "A44. touch head (headache)",
    "A45. touch chest (stomachache)", "A46. touch back (backache)", "A47. touch neck (neckache)", "A48. nausea or vomiting condition",
    "A49. use a fan (with hand or paper)", "A50. punching other person", "A51. kicking other person",
    "A52. pushing other person", "A53. pat on back of other person", "A54. point finger at the other person",
    "A55. hugging other person", "A56. giving something to other person", "A57. touch other person's pocket",
    "A58. handshaking", "A59. walking towards each other", "A60. walking apart from each other", "A61. put on headphone",
    "A62. take off headphone", "A63. shoot at the basket", "A64. bounce ball", "A65. tennis bat swing", "A66. juggling table tennis balls",
    "A67. hush (quite)", "A68. flick hair", "A69. thumb up", "A70. thumb down", "A71. make ok sign", "A72. make victory sign",
    "A73. staple book", "A74. counting money", "A75. cutting nails", "A76. cutting paper (using scissors)", "A77. snapping fingers",
    "A78. open bottle", "A79. sniff (smell)", "A80. squat down", "A81. toss a coin", "A82. fold paper", "A83. ball up paper",
    "A84. play magic cube", "A85. apply cream on face", "A86. apply cream on hand back", "A87. put on bag", "A88. take off bag",
    "A89. put something into a bag", "A90. take something out of a bag", "A91. open a box", "A92. move heavy objects", "A93. shake fist",
    "A94. throw up cap", "A95. hands up (both hands)", "A96. cross arms", "A97. arm circles", "A98. arm swings", "A99. running on the spot",
    "A100. butt kicks (kick backward)", "A101. cross toe touch", "A102. side kick", "A103. yawn", "A104. stretch oneself",
    "A105. blow nose", "A106. hit other person with something", "A107. wield knife towards other person",
    "A108. knock over other person (hit with body)", "A109. grab other person’s stuff", "A110. shoot at other person with a gun",
    "A111. step on foot", "A112. high-five", "A113. cheers and drink", "A114. carry something with other person",
    "A115. take a photo of other person", "A116. follow other person", "A117. whisper in other person’s ear",
    "A118. exchange things with other person", "A119. support somebody with hand", "A120. finger-guessing game (playing rock-paper-scissors)"
]


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, random_scale=False, random_mask=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param random_scale: scale skeleton length
        :param random_mask: mask some frames with zero
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.random_scale = random_scale
        self.random_mask = random_mask
        self.bone = bone
        self.vel = vel
        self.load_data()

        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C T V M   (output)
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.random_scale:
            data_numpy = tools.random_scale(data_numpy)
        if self.random_mask:
            data_numpy = tools.random_mask(data_numpy)
        # if self.bone:
        #     from .bone_pairs import ntu_pairs
        #     bone_data_numpy = np.zeros_like(data_numpy)
        #     for v1, v2 in ntu_pairs:
        #         bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        #     data_numpy = bone_data_numpy
        # if self.vel:
        #     data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        #     data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
