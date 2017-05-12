select earliest.bill_id, earliest.earliest_date, bv.bill_version_id as earliest_bvid from
(select bill_id, min(bill_version_action_date) as earliest_date from bill_version_tbl
group by bill_id) earliest
join bill_version_tbl bv on (earliest.bill_id=bv.bill_id and earliest.earliest_date=bv.bill_version_action_date)