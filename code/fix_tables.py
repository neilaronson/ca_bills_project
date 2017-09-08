import MySQLdb
import os
import codecs

def setup_db():
    """Setup connection and cursor to MySQL database based on environment variables"""
    global db
    db = MySQLdb.connect(host=os.environ["MYSQL_HOST"],
                         user=os.environ["MYSQL_USER"],
                         passwd=os.environ["MYSQL_PWD"],
                         db="capublic",
                         use_unicode=True,
                         charset='utf8')

    global cursor
    cursor = db.cursor()
    return db, cursor

def sql_try(command, fill_in):
    """Tries to execute a sql  command, prints if there's an error
    command: SQL command (string)
    fill_in: parameters (dictionary)
    """
    try:
        cursor.execute(command, fill_in)
        db.commit()
    except Exception as e:
        print e
        db.rollback()

def get_rows_to_change(table_name, id_col, file_col):
    """Finds all rows that need to be fixed by looking for rows that have a lob filename
    The lob filename means that the text in file_col should be replaced with the contents of that lob file
    table_name, id_col, file_col: strings
    """
    to_change = []
    try:
        select_statement = "SELECT {}, {} FROM {} WHERE {} LIKE '%.lob'".format(id_col, file_col, table_name, file_col)
        cursor.execute(select_statement)
        results = cursor.fetchall()
        for row in results:
            id_ = row[0]
            filename = row[1]
            to_change.append((id_, filename))
    except Exception as e:
        print e

    return to_change

def add_files(to_change, table_name, id_col, file_col):
    """Goes through rows that need to be changed and loads the appropriate lob file into the database
    in place of the filename.
    to_change: list of tuples to identify rows that need to be changed in format (id, file)
    table_name, id_col, file_col: strings
    """
    for row in to_change:
        id_ = row[0]
        filename = row[1]
        with codecs.open(filename, encoding='utf-8', errors='replace') as f:
            print "changing {}  based on file {}".format(id_, filename)
            file_text = f.readlines()
            file_text = "".join(file_text)
            add_file_sql = "UPDATE {} SET {}=%(file)s WHERE {}=%(id)s".format(table_name, file_col, id_col)
            fill_in = {'id': id_, 'file': file_text}
            sql_try(add_file_sql, fill_in)


def main():
    setup_db()
    all_tables_to_change = [
                # {'table': "veto_message_tbl", 'id_col': "bill_id", 'file_col': "message"},
                {'table': "bill_version_tbl", 'id_col': "bill_version_id", 'file_col': "bill_xml"}]
                # {'table': "law_section_tbl", 'id_col': "law_section_version_id", 'file_col': "content_xml"}]
    for item in all_tables_to_change:
        table = item['table']
        id_col = item['id_col']
        file_col = item['file_col']
        to_change = get_rows_to_change(table, id_col, file_col)
        add_files(to_change, table, id_col, file_col)
    db.close()

if __name__ == '__main__':
    main()
