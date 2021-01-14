until $(curl -o new_data.csv https://data.cdc.gov/api/views/y5bj-9g5w/rows.csv?accessType=DOWNLOAD&bom=true&format=true%20target=); do
    sleep 1
done

python excess_deaths.py > results.txt