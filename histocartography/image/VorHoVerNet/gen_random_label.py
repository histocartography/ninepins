from dataset import gen_pseudo_masks

for i in range(4, 7):
    gen_pseudo_masks(dataset='CoNSeP', split='train', ver=i, itr=0, contain_both=True)
    gen_pseudo_masks(dataset='CoNSeP', split='test', ver=i, itr=0, contain_both=True)