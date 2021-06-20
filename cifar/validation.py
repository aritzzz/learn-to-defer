from script_ours_copy import *
import json
import os
from collections import defaultdict

print("Validation...", flush=True)

os.makedirs('./Results/', exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bsz=1024

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
								 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
kwargs = {'num_workers': 1, 'pin_memory': True}

transform_test = transforms.Compose([
	transforms.ToTensor(),
	normalize
])
val_loader_cifar = torch.utils.data.DataLoader(
	datasets.__dict__["cifar10".upper()]('../data', train=False, transform=transform_test, download=True),
	batch_size=bsz, shuffle=True, **kwargs)

val_loader_svhn = torch.utils.data.DataLoader(
	datasets.SVHN('../data', download=True, split='test', transform=transform_test), batch_size=bsz, shuffle=True, **kwargs)


results = defaultdict(lambda:{})
for k in [2,3,5,6,8,9]:
	print("k {}".format(k), flush=True)
	expert = synth_expert(k, n_dataset)
	model = WideResNet(28, n_dataset + 1, 4, dropRate=0).to(device)
	model.load_state_dict(torch.load('checkpointK_'+str(k)+'.pt', map_location=device))
	result_cifar10=metrics_print(model, expert.predict, n_dataset, val_loader_cifar)
	results['cifar10'][k]=result_cifar10
	result_svhn=metrics_print(model, expert.predict, n_dataset, val_loader_svhn)
	results['svhn'][k]=result_svhn

with open('./Results/result.txt', 'w') as f:
	json.dump(results, f)
