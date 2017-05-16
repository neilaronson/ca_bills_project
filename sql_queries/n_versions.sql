select b.bill_id, count(bill_version_id) as n_versions, passed from bill_version_tbl bv
join bill_tbl b on b.bill_id=bv.bill_id
group by b.bill_id