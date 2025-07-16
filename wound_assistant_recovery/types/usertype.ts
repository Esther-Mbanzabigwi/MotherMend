export type UserRole = "authenticated" | "admin" | "user";

export interface IUser {
  id: number;
  documentId: string;
  username: string;
  email: string;
  phone?: string;
  createdAt: string;
}

export interface ILogInUser {
  identifier: string;
  password: string;
}

export interface IRegisterUser {
  username: string;
  email: string;
  phone: string;
  password: string;
  profile?: File;
}

export interface ILoginResponse {
  jwt: string;
  user: IUser;
}
