import sys
sys.path.append('/data/hongdeyao/code/RRec')
from datasets import load_from_disk
dataset = load_from_disk('/data/hongdeyao/Musical_Instruments_0_2022-10-2023-10')
test_data = dataset['test']
count = 0
for i, sample in enumerate(test_data):
    if sample.get('user_id') == 'AEAMP2QKN2BR7IDXXLELLPWEDV4Q':
        count += 1
        print(f'Sample {i} (count={count}):')
        print('history_item_id:', sample['history_item_id'])
        print('item_id:', sample['item_id'])
        print('history_item_title:', sample['history_item_title'])
        print('history_rating:', sample['history_rating'])
        print('---')