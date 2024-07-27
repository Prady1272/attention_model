# for category in "Display" "Oven" ; do
#   # python train.py --model "transformer" --data_root "snippets/data/partnet_root"  --pretraining --category "$category" > "transformer_l1_all_pretraining/$category.txt"
#   python train.py --model "transformer" --lr 0.001 --num_layers 4 --category "$category" > "transformer_l1_all_fine_tuning/$category.txt"

# done



for category in  "Refrigerator" "Display" "Laptop" "Knife" "Clock"  "Scissors" "Door" ; do
  python train.py --model "transformer" --data_root "snippets/data/partnet_root"  --pretraining --category "$category" > "transformer_l1_all_pretraining/$category.txt"
  python train.py --model "transformer" --category "$category" > "transformer_l1_all_fine_tuning/$category.txt"

done

# for category in  "Bottle" "Refrigerator" "Display" "Laptop" "Knife" "Clock"  "Scissors" "Door" ; do
#   python train.py --model "transformer" --category "$category" > "transformer_l1_all_fine_tuning/$category.txt"
# done



# "Bottle" "Refrigerator" "Display" "Laptop" "Knife" "Clock"  "Scissors" "Door" 
# for category in "Bottle" "Refrigerator" "Display" "Laptop" "Knife" "Clock"  "Scissors" "Door" "Pen" "Pliers" "Oven" "Cart" "USB" ; do
#   python train.py --model "transformer" --encode_shape=False --category "$category" > "transformer_l1_wse/$category.txt"
# done

# for category in "Bottle" "Refrigerator" "Display" "Laptop" "Knife" "Clock"  "Scissors" "Door" "Pen" "Pliers" "Oven" "Cart" "USB" ; do
#   python train.py --model "transformer" --encode_part=False --category "$category" > "transformer_l1_wpe/$category.txt"
# done

# for category in "Bottle" "Refrigerator" "Display" "Laptop" "Knife" "Clock"  "Scissors" "Door" "Pen" "Pliers" "Oven" "Cart" "USB" ; do
#   python train.py --model "transformer" --encode_part=False --encode_shape=False --category "$category" > "transformer_l1_wpe_wse/$category.txt"
# done


# currently working
# "Bottle" "Refrigerator" "Display" "Laptop" "Knife" "Clock"  "Scissors" "Door" "Pen" "Pliers" "Oven" "Cart" "USB"

# easy+hard
# "Bottle" "Refrigerator" "Display" "Laptop" "Knife" "Clock"  "Scissors" "StorageFurniture" "Door" "Pen" "Pliers" "Oven" "Cart" "USB"

# easy
#  "Bottle" "Refrigerator" "Laptop" "Scissors" 


# hard
# "Display" "Knife" "Clock" "StorageFurniture" "Door" "Pen" "Pliers" "Oven" "Cart" "USB"

# for category in "Refrigerator" "Display" "Laptop" "Knife" "Clock" "Bottle" "Scissors" "Table" "Dishwasher" "Storage_furniture" "Door_set" "Pliers" "Pen" "Stapler" "Oven" "Luggage" "Window" "Cart" "USB" "FoldingChair"; do
#   python train.py  --category "$category" > "hi/$category.txt"
# done