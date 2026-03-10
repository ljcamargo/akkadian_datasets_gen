echo "All Datasets"
find workspace/outputs -xdev -name "*.csv" -print0 | du -hcx --files0-from=- | tail -n 1
find workspace/outputs -name "*.csv" -exec wc -l {} +

echo "Pretrain"
find workspace/outputs -xdev -name "*pretrain.csv" -print0 | du -hcx --files0-from=- | tail -n 1
find workspace/outputs -name "*pretrain.csv" -exec wc -l {} +

echo "Finetunes"
find workspace/outputs -xdev -name "*finetune.csv" -print0 | du -hcx --files0-from=- | tail -n 1
find workspace/outputs -name "*finetune.csv" -exec wc -l {} +

