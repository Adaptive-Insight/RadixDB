insert into tokens(context, name, value)
  values ('bot1', 'slack_signing_secret', '9d334074da13bfb5bc5c362cba9b2a07'),
         ('bot1', 'slack_bot_token', 'xoxb-309648771139-1019504891490-IvzC2ErOII1k9TxdmJ6XNPBn');
create table hist_prices (
    sym text,
    date timestamp with time zone,
    close double precision,
    high double precision,
    low  double precision,
    open double precision,
    volume bigint,
    adjClose double precision,
    adjHigh double precision,
    adjLow double precision,
    adjOpen double precision,
    adjVolume bigint,
    divCash double precision,
    splitFactor double precision);

create table animals as
    SELECT * FROM (VALUES ('pig', 1.5, 0.7),
                          ('rabbit', 0.5, 0.2),
                          ('duck', 1.2, 0.15),
                          ('chicken', 0.9, 0.2),
                          ('horse', 3, 1.1)
      ) AS t (name, length, width);
