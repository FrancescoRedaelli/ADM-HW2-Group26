ids=$(awk -F, 'length($3)>100' instagram_posts.csv | cut -f  4 | head -10)
for term in $ids; do grep $term instagram_profiles.csv || echo 'User not found!'; done