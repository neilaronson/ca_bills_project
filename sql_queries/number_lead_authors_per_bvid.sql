SELECT bill_version_id, count(*) FROM capublic.bill_version_authors_tbl
where contribution='LEAD_AUTHOR'
group by bill_version_id
having count(*)>1;