-- SQLite
-- SELECT * 
-- FROM sales_data;

-- SELECT *
-- FROM sales_data
-- WHERE store_id = '112'
-- AND prod_cd = '98050138';

select DISTINCT store_id, bsns_dt
from sales_data
where store_id = '515';