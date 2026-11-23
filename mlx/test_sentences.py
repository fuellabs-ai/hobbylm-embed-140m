"""The 6 test inputs: 3 queries, 3 passages. One very short (2 words), one long (about 800 tokens before truncation)."""

LONG_PASSAGE = (
    "The city library system runs twelve branches, a central archive and a mobile van that visits outlying "
    "villages twice a week. Each branch keeps its own opening hours, which are set every spring after a review "
    "of visitor counts, staff availability and requests from local schools. The central branch opens at eight "
    "in the morning on weekdays and stays open until nine at night, while smaller branches usually open at ten "
    "and close at six. On weekends most branches open only on Saturday, and the central branch is the only one "
    "that opens on Sunday afternoons. Members can borrow up to twenty items at a time across all branches. "
    "Books are lent for three weeks, films and music for one week, and reference works cannot leave the building. "
    "Loans can be renewed online, by phone or at any desk, as long as nobody else has placed a hold on the item. "
    "An item can be renewed up to three times, after which it has to be returned and borrowed again. Late fees "
    "were removed for children's material several years ago, and the board later extended this to adult "
    "material as well, because the fees cost more to collect than they brought in. Lost or damaged items are "
    "still charged at the replacement price, and a member with more than a set amount owed cannot borrow until "
    "the balance is paid or a payment plan is agreed. Joining is free for anyone who lives, works or studies in "
    "the city. New members need a photo identity document and proof of address, such as a utility bill or a "
    "tenancy agreement dated within the last three months. Students can use a letter from their school or "
    "college instead. Children under twelve join with a parent or guardian, who signs the form and takes "
    "responsibility for the child's loans. Visitors from outside the city can buy a yearly card at a small fee, "
    "which gives the same borrowing rights as a full member. The digital collection includes e-books, audio "
    "books, newspapers and several language courses. These are reached through the library website or a "
    "separate reading app, using the same card number and a four digit PIN chosen at sign up. A forgotten PIN "
    "can be reset at any branch desk, or through the website by answering the security question set when the "
    "account was created. If the security question has also been forgotten, the member must visit a branch "
    "with their photo identity document. Every branch has free wireless internet and public computers that can "
    "be booked for one hour at a time, with a second hour allowed if nobody is waiting. Printing and scanning "
    "are available at a small charge per page, paid by card or with credit added to the library account. The "
    "central branch also has a quiet study floor, six group rooms that can be booked up to two weeks ahead, and "
    "a makerspace with a 3D printer, a laser cutter and sewing machines. Using the makerspace tools needs a "
    "short safety session first, which runs every Tuesday evening and Saturday morning. The archive holds old "
    "maps, local newspapers going back more than a century, parish records and photographs donated by "
    "residents. Most of it can be viewed by appointment in the reading room, where bags and pens are not "
    "allowed and pencils are provided. Staff can scan pages for researchers who cannot visit in person, with "
    "a fee that depends on the size and condition of the original. Events run all year: story time for toddlers "
    "on weekday mornings, homework clubs after school, book groups for adults, and a summer reading challenge "
    "that last year drew several thousand children. Most events are free, but some workshops ask for a small "
    "fee to cover materials, and places should be booked in advance because they fill quickly. Accessibility "
    "is a priority across the system. All branches have step free entrances, and the larger ones have hearing "
    "loops at the main desk and large print and dyslexia friendly collections. A home delivery service brings "
    "books to members who cannot leave their homes because of illness, disability or caring duties. Volunteers "
    "help run this service and are trained and checked before they start. Feedback can be left at any desk, "
    "through the website form or by email, and the library publishes a summary of comments and its responses "
    "every quarter. Complaints are answered within ten working days, and anyone who is not happy with the "
    "answer can ask for it to be reviewed by a senior manager who was not involved in the original reply."
)

SENTENCES = [
    {"name": "q_short", "prompt": "query", "text": "solar panels"},
    {"name": "q_capital", "prompt": "query", "text": "what is the capital of france"},
    {"name": "q_password", "prompt": "query", "text": "how do I reset my PIN for the library app"},
    {"name": "p_paris", "prompt": "passage", "text": "Paris is the capital and largest city of France."},
    {"name": "p_reset", "prompt": "passage", "text": (
        "To reset your password, open the app, tap Settings, then Account, and choose Reset password. "
        "A link will be sent to the email address on your account.")},
    {"name": "p_long", "prompt": "passage", "text": LONG_PASSAGE},
]
