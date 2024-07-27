
for category in "Display" "Laptop" "Knife" "Clock"  "Scissors" "Door" "Pen" "Pliers" "Oven" "Cart" "USB" ; do
  python train.py --model "transformer" --category "$category" > "transformer_l1_split2_layer_norm/$category.txt"
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