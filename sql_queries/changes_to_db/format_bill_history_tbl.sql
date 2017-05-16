ALTER TABLE `capublic`.`bill_history_tbl`
ADD COLUMN `session_year` VARCHAR(8) NULL DEFAULT NULL AFTER `end_status`;
UPDATE bill_history_tbl SET session_year=SUBSTRING(bill_id,1,8);
ALTER TABLE `capublic`.`bill_history_tbl`
ADD COLUMN `session_num` VARCHAR(2) NULL DEFAULT NULL AFTER `session_year`;
UPDATE bill_history_tbl SET session_num=SUBSTRING(bill_id,9,1);
