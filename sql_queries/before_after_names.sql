select introduced.bill_id, passed.bill_id, introduced.names, passed.names
from (select bv.bill_id, group_concat(bva.name) as names from bill_version_authors_tbl bva
join legislator_tbl l on bva.name=l.author_name and l.session_year=bva.session_year
join bill_version_tbl bv on bv.bill_version_id=bva.bill_version_id
where contribution='LEAD_AUTHOR' and bv.bill_version_id like '%INT'
group by bv.bill_id) introduced
right join (select b.bill_id, group_concat(bva.name) as names from bill_tbl b
join bill_version_authors_tbl bva on b.latest_bill_version_id=bva.bill_version_id
where passed=1 and measure_type in ('AB', 'SB')
group by b.bill_id) passed
on introduced.bill_id=passed.bill_id
