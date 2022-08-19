db = [
    {"title": "Moby Dick", "author": "Herman Melville", "release date": 1851},
    {"title": "A Study in Scarlet", "author": "Sir Arthur Conan Doyle", "release date": 1887},
    {"title": "Hitchhikers Guide to the Galaxy", "author": "Douglas Adams", "release date": 1879},
]

def locate_by_title(db, title):
    for book in db:
        if book["title"] == title:
            return book


print(locate_by_title(db, "Moby Dick"))


def books_released_after(db, year):
    released_after = []
    for book in db:
        released = book["release date"]
        if released > year:
            released_after.append(book)
    return released_after


print(books_released_after(db, 1850))


def update(db, key, value, where_key, where_value):
    for book in db:
        if book[where_key] == where_value:
            book[key] = value


update(db, key="release year", value=1979, where_key="title", where_value="Hitchhikers Guide to the Galaxy")
print(db)


def query(db, where_key, where_value, where_qualifier):
    results = []
    for book in db:
        if where_qualifier == "exactly" and book[where_key] == where_value:
            results.append(book)
        elif where_qualifier == "greater than" and book[where_key] > where_value:
            results.append(book)
        elif where_qualifier == "less than" and book[where_key] < where_value:
            results.append(book)
    return results


results = query(db,
                where_key="title",
                where_value="Moby Dick",
                where_qualifier="exactly")

print(results)


results = query(db,
                where_key="release date",
                where_value=1850,
                where_qualifier="greater than")

print(results)

