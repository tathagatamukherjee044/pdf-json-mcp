const sample = {
    "templateConfig": {
      "delimiter": " ",
      "stopWords": "_________",
      "invoiceLine": 1, // Number of lines to be skipped before the invoice table
      "columnLength": 3, // Number of columns in the invoice table
    },
    "templateNm": "CUSTOM#2_PDF", // Name of the template not required
    "footer": [ // Shows the end of invoice table
      "Sum Total",
      "_________",
      "RELIANCE RETAIL LIMITED 712",
      "Payment Advice",
      "Payment document"
    ],
    "header": [ // Shows the start of invoice table
      "Doc no",
      "_________"
    ],
    "remittanceConfig": [ // Remittance configuration (ie.Payment Data, outside the table)
      {
        "token": "Pay Number", //Token found outside table this corresponds to the UTR number
        "delimiter": " ", // Delimiter for the table headers (usually space)
        "position": 1, // Position of the token in the invoice (starts from 0, and -1 means last column)
        "stdNm": "PABankUtr" // Standard name of the token (our standard name for the token)
      },
      {
        "token": "Date", //Token found outside table this corresponds to the payment date
        "delimiter": " ", // Delimiter for the table headers (usually space)
        "position": 1, // Position of the token in the invoice (starts from 0, and -1 means last column)
        "format": "dd/MM/yyyy", // Format of the token (usually dd/MM/yyyy, if stdName is payDate then ont this is required)
        "stdNm": "payDate" // Standard name of the token (our standard name for the token)
      },
    ],
    "docConfig": [ // Document configuration (Invoice Data ie.inside a table)
      {
        "token": "Doc no", // Token to be found in the invoice (ie.Invoice Data)
        "delimiter": " ", // Delimiter for the table headers (usually space)
        "position": 1, // Position of the token in the invoice (starts from 0, and -1 means last column)
        "stdNm": "docId" // Standard name of the token (our standard name for the token)
      },
      {
        "token": "Amount", // Token to be found in the invoice (ie.Invoice Data)
        "delimiter": " ", // Delimiter for the table headers (usually space)
        "position": 2, // Position of the token in the invoice (starts from 0, and -1 means last column)
        "stdNm": "docAmt" // Standard name of the token (our standard name for the token)
      },
    ]
  }