ALTER TABLE `capublic`.`bill_version_authors_tbl` 
ADD COLUMN `session_year` VARCHAR(45) NULL DEFAULT NULL AFTER `primary_author_flg`;
UPDATE bill_version_authors_tbl
	JOIN bill_version_tbl bv on bill_version_authors_tbl.bill_version_id=bv.bill_version_id
SET bill_version_authors_tbl.session_year=SUBSTRING(bv.bill_id,1,8);
ALTER TABLE `capublic`.`bill_version_authors_tbl`
ADD COLUMN `bill_id` VARCHAR(45) NULL DEFAULT NULL AFTER `session_year`;
UPDATE bill_version_authors_tbl
	JOIN bill_version_tbl bv on bill_version_authors_tbl.bill_version_id=bv.bill_version_id
SET bill_version_authors_tbl.bill_id=bv.bill_id;
