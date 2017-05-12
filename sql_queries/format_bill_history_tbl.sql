ALTER TABLE `capublic`.`bill_history_tbl` 
ADD COLUMN `session_year` VARCHAR(8) NULL DEFAULT NULL AFTER `end_status`;
UPDATE bill_history_tbl SET session_year=SUBSTRING(bill_id,1,8);