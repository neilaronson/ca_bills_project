ALTER TABLE `capublic`.`bill_tbl` 
ADD COLUMN `passed` INT NULL DEFAULT NULL AFTER `days_31st_in_print`;
update bill_tbl set passed=case when measure_state in ('Chaptered','Enrolled') then 1 else 0 end;

