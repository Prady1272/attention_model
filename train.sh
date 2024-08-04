

for split_index in 0 1 2; do
    for category_index in 0; do 
        python -u train.py   --use_multi_loss  --split_index $split_index --category_index $category_index --num_layers 6 --use_seed > "$category_index"_"$split_index".txt
    done
done
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