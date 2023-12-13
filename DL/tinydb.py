from tinydb import TinyDB, Query

if __name__ == '__main__':
    db = TinyDB('db.json')
    query = Query()

    def insert():
        db.insert({'paramA': 1, 'paramB': 2})
        db.insert({'paramA': 1, 'paramB': 4})
        db.insert({'paramA': 3, 'paramB': 2})
        db.insert({'paramA': 3, 'paramB': 4})

    # db.purge()
    print(db) 
    insert()
    print(db)
    print(db.all())

    def search():
        return db.search(query.paramA == 1)

    results = search()
    print(results)

    def update():
        results = db.search(query.paramB == 4)
        for res in results:
            res['paramA'] = 5
        db.write_back(results)

    update()
    print(db.all())


    def delete():
        db.remove(query.paramA == 5)

    delete()
    print(db.all())








