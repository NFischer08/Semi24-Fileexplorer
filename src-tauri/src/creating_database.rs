use serde::{Deserialize, Serialize};
use surrealdb::engine::any::connect;
use surrealdb::opt::auth::Root;
use surrealdb::RecordId;

#[derive(Debug, Serialize, Deserialize)]
struct Person {
    name: Option<String>,
    marketing: Option<bool>
}

#[derive(Debug, Deserialize)]
struct Record {
    id: RecordId,
}
#[tokio::main]
async fn main() -> surrealdb::Result<()> {
    let db = connect("ws://localhost:8000").await?;
    db.signin(Root {
        username: "root",
        password: "root",
    })
        .await?;
    db.use_ns("ns").use_db("db").await?;

    // Create a record with a random ID
    let person: Option<Person> = db.create("person").await?;
    dbg!(person);
    // Create a record with a specific ID
    let record: Option<Record> = db
        .create(("person", "tobie"))
        .content(Person {
            name: Some("Tobie".into()),
            marketing: Some(true),
        })
        .await?;
    dbg!(record);
    Ok(())
}